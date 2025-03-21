import requests
from functools import wraps
from typing import Dict, Callable, Any

def url_parser(url: str, headers: Dict = None, **default_params):
    """
    URL 요청 데코레이터 (파라미터 유연하게 설정 가능)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            final_headers = headers or {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/134.0.0.0 Safari/537.36"
                )
            }

            # kwargs에 `params`가 있으면 오버라이딩, 없으면 기본값 사용
            final_params = kwargs.pop("params", default_params)

            try:
                response = requests.get(url, headers=final_headers, params=final_params, timeout=10)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                raise Exception(f"❌ 요청 실패: {e}")

            return func(response, *args, **kwargs)

        return wrapper
    return decorator
