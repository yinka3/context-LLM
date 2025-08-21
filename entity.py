from typing import List, Union


class EntityPipeline:
    def __init__(self, document: Union[List[str], str]):
        self.doc = document
        