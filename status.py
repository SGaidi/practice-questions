from typing import List, Tuple


def read_csv() -> List[Tuple]:
	rows = []
	with open('practice_questions.csv', 'r') as f:
		lines = f.readlines()
	for line in lines:
		name, done, total = line[:-1].split(',')
		done, total = int(done), int(total)
		percent = round(done / total, 3)
		rows.append((name, done, total, percent))
	return rows
	

def sorted_percent(rows: List[Tuple]) -> List[Tuple]:
	return sorted(rows, key=lambda record: record[3])
	
	
def rate_of(rows: List[Tuple], days_left: int):
	return sum(record[2]-record[1] for record in rows) / days_left
	
	
def percent_of(rows: List[Tuple]) -> int:
	return 100 * sum(
		record[1] for record in rows) / sum(record[2] for record in rows)
	
	
data = read_csv()
rate = rate_of(data, 60)
percent = percent_of(data)
print('Category,done,total,percent')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('\n'.join(str(item) for item in sorted_percent(data)))
print(f'daily rate to finish all questions in 2 months: {rate:.2f}')
print(f'{percent:.2f}% of questions completed so far.')
