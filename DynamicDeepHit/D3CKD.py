import json

class DomainInformedModel:

    def __init__(self):
        with open('config.json') as json_data:
            self.config = json.load(json_data)


    def main(self):
        """Main call function"""
        print(self.config)