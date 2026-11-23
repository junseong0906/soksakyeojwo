from dataclasses import dataclass, field


LOCAL_HOST = '127.0.0.1'
HEALTH_PATH = '/soksakyeojwo/health'


@dataclass(frozen=True)
class LocalEngineEndpoint:
    port: int
    access_key: str = field(repr=False)

    def __post_init__(self):
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError('로컬 엔진 포트가 올바르지 않습니다.')
        if not self.access_key:
            raise ValueError('로컬 엔진 연결 키가 없습니다.')

    @property
    def asr_url(self):
        return f'ws://{LOCAL_HOST}:{self.port}/asr?mode=full'

    @property
    def health_url(self):
        return f'http://{LOCAL_HOST}:{self.port}{HEALTH_PATH}'

    @property
    def headers(self):
        return {'Authorization': f'Bearer {self.access_key}'}
