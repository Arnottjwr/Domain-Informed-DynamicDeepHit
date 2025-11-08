import csv

data = list(csv.DictReader(open('dummy_timeseries_data.csv')))

print('=== SAMPLE DATA FROM MULTIPLE PATIENTS ===\n')

patients = sorted(set(r['subject_id'] for r in data))[:3]
for pid in patients:
    rows = [r for r in data if r['subject_id'] == pid]
    print(f'Patient {pid} ({rows[0]["event"]}, {rows[0]["gender"]}):')
    print(f'  Measurements: {len(rows)}')
    print(f'  First date: {rows[0]["chartdate"]}')
    print(f'  Last date: {rows[-1]["chartdate"]}')
    print(f'  Event time: {rows[-1]["event_time"]}')
    print(f'  Age range: {rows[0]["age_calculated"]} to {rows[-1]["age_calculated"]}')
    print(f'  Years value: {rows[0]["year"]}')
    print()
