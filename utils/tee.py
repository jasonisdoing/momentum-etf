"""STDOUT과 파일에 동시에 쓰기 위한 헬퍼 클래스입니다."""


class Tee:
    """STDOUT과 파일에 동시에 쓰기 위한 헬퍼 클래스입니다."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()
