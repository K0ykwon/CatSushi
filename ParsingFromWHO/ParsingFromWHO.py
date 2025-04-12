import json

class ParsingFromWHO:
    def __init__(self, file_path='ParsingFromWHO/dosing.json'):
        self.file_path = file_path
        self.dosing_data = self._load_dosing_data()

    def __call__(self, antibiotic_name):
        return self.find_dosing_by_antibiotic(antibiotic_name)
    def _load_dosing_data(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from '{self.file_path}'.")
            return []

    def find_dosing_by_antibiotic(self, name):
        for entry in self.dosing_data:
            if entry['antibiotic'].lower() == name.lower():
                return entry
        return None


