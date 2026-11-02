"""Execute only the prepared notebook's watcher AST with fake child/files/time."""
import ast
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import build_yash_nov2_notebooks as notebook


def counts():
    return dict(poll_attempts=0, read_errors=0, errors_by_type={}, warnings_emitted=0,
                warnings_suppressed=0, warning_output_errors=0, last_error_type=None)


def watcher_parts():
    tree=ast.parse(notebook.RUN_AND_EXPORT)
    helper=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="read_status_text")
    loop=next(node for node in ast.walk(tree) if isinstance(node,ast.While) and ast.unparse(node.test)=="process.poll() is None")
    return helper,loop


class FakeStudy:
    def __init__(self,path):self.path=path
    def __truediv__(self,name):
        assert name=="status.json"
        return self.path


class NotebookWatcherTests(unittest.TestCase):
    def namespace(self,path,child,remaining=lambda:100):
        helper,loop=watcher_parts()
        monitor=counts()
        namespace={"process":child,"STUDY":FakeStudy(path),"monitor_counts":monitor,
            "remaining_seconds":remaining,"hard_stop_at":1000.,"previous":None,"last_print":0.,
            "time":SimpleNamespace(monotonic=lambda:1.,sleep=Mock()),"console":"synthetic-console",
            "interrupt_owned":Mock(return_value="synthetic_sigint"),"run_record":{"status":"started","monitor":monitor}}
        exec(compile(ast.Module(body=[helper],type_ignores=[]),"watcher-helper","exec"),namespace)
        return namespace,loop

    def test_exists_read_race_and_temporary_io_never_interrupt_healthy_child(self):
        path=Mock()
        path.exists.side_effect=AssertionError("No check-then-read filesystem race is allowed")
        path.read_text.side_effect=[FileNotFoundError("atomic replacement gap"),OSError("temporary Drive IO"),'{"status":"running"}']
        child=Mock()
        child.poll.side_effect=[None,None,None,0]
        namespace,loop=self.namespace(path,child)
        with patch("builtins.print"):
            exec(compile(ast.Module(body=[loop],type_ignores=[]),"actual-notebook-watch-loop","exec"),namespace)
        namespace["interrupt_owned"].assert_not_called()
        child.send_signal.assert_not_called()
        path.exists.assert_not_called()
        self.assertEqual(namespace["previous"],'{"status":"running"}')
        monitor=namespace["run_record"]["monitor"]
        self.assertEqual(monitor["errors_by_type"],{"FileNotFoundError":1,"OSError":1})
        self.assertEqual(monitor["poll_attempts"],3)
        self.assertEqual(json.loads(json.dumps(namespace["run_record"]))["monitor"]["read_errors"],2)

    def test_status_warnings_are_bounded_while_all_errors_are_counted(self):
        path=Mock();path.read_text.side_effect=PermissionError("temporary Drive issue")
        namespace,_=self.namespace(path,Mock())
        with patch("builtins.print") as output:
            for _ in range(100):self.assertIsNone(namespace["read_status_text"](path,namespace["monitor_counts"]))
        self.assertEqual(output.call_count,5)
        self.assertEqual(namespace["monitor_counts"]["read_errors"],100)
        self.assertEqual(namespace["monitor_counts"]["warnings_suppressed"],95)

    def test_deadline_still_interrupts_owned_child_before_optional_status_read(self):
        path,child=Mock(),Mock();child.poll.return_value=None
        namespace,loop=self.namespace(path,child,remaining=lambda:0)
        exec(compile(ast.Module(body=[loop],type_ignores=[]),"actual-notebook-deadline-loop","exec"),namespace)
        namespace["interrupt_owned"].assert_called_once_with(child)
        path.read_text.assert_not_called()
        self.assertEqual(namespace["run_record"]["status"],"deadline_or_27000_second_stop_requested")

    def test_prepared_deadline_is_0850_and_source_budget_is_unchanged(self):
        tree=ast.parse(notebook.BOOTSTRAP)
        deadline=next(node.value for node in tree.body if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id=="DEADLINE" for target in node.targets))
        self.assertEqual([ast.literal_eval(value) for value in deadline.args], [2026,11,2,8,50])
        self.assertNotIn("09:20",notebook.BOOTSTRAP+notebook.RUN_AND_EXPORT)
        self.assertIn("min(27000.,remaining)",notebook.RUN_AND_EXPORT)
        compile(notebook.RUN_AND_EXPORT,"prepared-run-cell","exec")


if __name__=="__main__":unittest.main()
