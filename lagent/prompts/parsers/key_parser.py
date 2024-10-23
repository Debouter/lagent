from typing import List
from lagent.prompts.parsers import StrParser

# 基于关键词匹配解析response
class KeyParser(StrParser):

    def __init__(
        self,
        keywords: List[str],
        template: str = "",
        **kwargs,
    ):
        super().__init__(template, **kwargs)
        self.keywords = keywords
    
    def parse_response(self, data: str) -> dict:
        last_keyword = "Invalid response: No keywords found"
        last_index = -1
        for keyword in self.keywords:
            index = data.rfind(keyword) 
            if index > last_index:  
                last_keyword = keyword
                last_index = index
        return dict(
            status = last_keyword
        )


if __name__ == "__main__":
    keyparser = KeyParser(keywords=["**COMPLETE**", "**INCOMPLETE**"])
    data = "The step is **COMPLETE**"
    print(keyparser.parse_response(data))
    