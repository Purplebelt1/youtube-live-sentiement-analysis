class LiveChatMessage:
    def __init__(self, text_content, vader_output, roberta_output, cleaned_tm_list = None, cleaned_message=None, tm_output=None):
        self._text_content = text_content
        self._vader_output = vader_output
        self._roberta_output = roberta_output
        self._cleaned_tm_list = cleaned_tm_list
        self._tm_output = tm_output
        self._cleaned_message = cleaned_message

    @property
    def cleaned_message(self):
        return self._cleaned_message

    @cleaned_message.setter
    def cleaned_message(self, value):
        self._cleaned_message = value

    @property
    def roberta_output(self):
        return self._roberta_output

    @roberta_output.setter
    def roberta_output(self, value):
        self._roberta_output = value

    @property
    def text_content(self):
        return self._text_content

    @text_content.setter
    def text_content(self, value):
        self._text_content = value

    @property
    def vader_output(self):
        return self._vader_output

    @vader_output.setter
    def vader_output(self, value):
        self._vader_output = value

    @property
    def cleaned_tm_list(self):
        return self._cleaned_tm_list

    @cleaned_tm_list.setter
    def cleaned_tm_list(self, value):
        self._cleaned_tm_list = value

    @property
    def tm_output(self):
        return self._tm_output

    @tm_output.setter
    def tm_output(self, value):
        self._tm_output = value