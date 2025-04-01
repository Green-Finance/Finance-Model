from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

class WebSearch:
    def __init__(self, region="ko-kr", time="d", max_results=5):
        self.wrapper = DuckDuckGoSearchAPIWrapper(
            region=region,
            time=time,
            max_results=max_results
        )
        self.search = DuckDuckGoSearchResults(
            api_wrapper=self.wrapper,
            source="news",
            output_format="list"
        )
