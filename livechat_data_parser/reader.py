
import json
import os
import pandas as pd

def load_dataframe(file_name):

    with open(file_name, 'r', encoding='utf-8') as file:
        messages = json.load(file)

    dataframe = pd.json_normalize(messages)

    dataframe = dataframe[dataframe["message_type"] == "Chat Message"]

    columns_to_drop = ["content.emoji",
                    "content.membershipRenewal",
                    "content.purchaseAmount.simpleText",
                    "content.membershipChat",
                    "content.purchase_amount",
                    "content.sticker_description",
                    "content.membershipJoin",
                    "message_type"]

    for column in columns_to_drop:
        if column in dataframe.columns:
            dataframe = dataframe.drop(columns=[column])

    return dataframe

if __name__ == "__main__":
    load_dataframe("\\json_test.json")