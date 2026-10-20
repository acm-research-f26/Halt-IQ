"""Run/resume the time-bounded 150-question real study, then train/evaluate all seeds."""
import study
study.STUDY = study.ROOT / 'study150'
if __name__ == '__main__':
    study.generate('http://127.0.0.1:8080/v1/chat/completions', None)
    study.run()
