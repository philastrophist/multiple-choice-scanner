from datetime import datetime
from pathlib import Path
import questionary


def save_calibration():
    pass

def load_calibration():
    pass

def save_answer_key():
    pass

def load_answer_key():
    pass

def save_student():
    pass

def load_student():
    pass

def mark_paper(calibration, answer, student, store: Path):
    pass

def read_crewcode(paper):
    pass

def read_title(paper):
    pass

def make_calibration_sheets():
    """
    return title of the qa, sections
    sections: ordereddict(image1=sections, image2=sections)
    """
    return title, sections

if __name__ == '__main__':
    answer = questionary.select("What do you want to do?", ["Start marking", "Open result"]).ask()
    if answer == "Start marking":
        print("start scanning empty sheets...")
        title, sections = make_calibration_sheets()
        save_calibration(title, sections)
        print("start scanning answer key...")
        make_




    # today = datetime.today()
    # datetime.strftime(today, f'YYYY-MM-DD-{}')
    # session_name =