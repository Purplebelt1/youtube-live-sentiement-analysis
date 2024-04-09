"""module for a livechat parser that takes a raw response json from youtube and will
parse out the relevant chat data that is needed for the output."""
import json

from bs4 import BeautifulSoup


class LivechatParser:
    """LivechatParser class takes a raw response from youtube and pulls out relevant content
    for data we are looking for(livechat info, superchats, memberships)"""
    parse_type = ""
    soup_parser = None
    livechat_contents = None
    initial_continuation = None

    def __init__(self, parse_type):
        self.parse_type = parse_type

    def build_parser(self, livechat_data):
        """returns a BeautifulSoup parser for content extraction."""
        self.soup_parser = BeautifulSoup(livechat_data.text, self.parse_type)

    def find_content(self):
        """parser seeks out the content where the livechat data for output resides
        in the raw response"""
        script = next(x for x in self.soup_parser.find_all('script') if "authorName" in x.text)
        content = script.text

        start_index = content.find('{')
        end_index = len(content) - content[::-1].find('}')
        livechat_data = json.loads(content[start_index:end_index])

        header_renderer = \
        livechat_data["continuationContents"]["liveChatContinuation"]["header"]["liveChatHeaderRenderer"][
            "viewSelector"]["sortFilterSubMenuRenderer"]["subMenuItems"][1]["continuation"]["reloadContinuationData"][
            "continuation"]
        self.initial_continuation = header_renderer

        return livechat_data["continuationContents"]["liveChatContinuation"]["actions"]

