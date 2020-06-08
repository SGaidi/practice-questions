import typing
from typing import List, Tuple


def read_db() -> List[Tuple]:
	data = []
	with open('practice_questions.csv', 'r') as f:
		lines = f.readlines()
	for line in lines:
		name, done, total = line[:-1].split(',')
		done, total = int(done), int(total)
		precent = round(done / total, 3)
		data.append((name, done, total, precent))
	return data
	

def sorted_precent(db: List[Tuple]) -> List[Tuple]:
	return sorted(db, key=lambda record: record[3])
	
	
def rate_of(db: List[Tuple], days_left: int):
	return sum(record[2]-record[1] for record in db) / days_left
	
	
def precent_of(db: List[Tuple]) -> int:
	return 100 * sum(record[1] for record in db) / sum(record[2] for record in db)
	
	
db = read_db()
rate = rate_of(db, 365)
precent = precent_of(db)
print('\n'.join(str(item) for item in sorted_precent(db)))
print(rate)
print(precent)