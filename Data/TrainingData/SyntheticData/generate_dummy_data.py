import csv
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)

def generate_patient_data(patient_id, num_measurements):
    """Generate time series data for a single patient"""

    # Patient-level constants (don't change within patient)
    subject_id = 10000000 + patient_id  # 8-digit ID
    gender = random.choice(['M', 'F'])
    initial_age = random.randint(25, 85)
    event = random.choice(['alive', 'alive', 'alive', 'alive', 'dead', 'dialysis'])  # 75% alive, 25% dead

    # Generate first measurement date (random date in past 5 years)
    days_back = random.randint(365, 365 * 5)
    first_date = datetime.now() - timedelta(days=days_back)

    # Generate measurement dates (sorted chronologically)
    measurement_dates = [first_date]
    current_date = first_date

    for _ in range(num_measurements - 1):
        # Random interval between measurements (30-180 days)
        days_forward = random.randint(30, 180)
        current_date = current_date + timedelta(days=days_forward)
        measurement_dates.append(current_date)

    last_measurement_date = measurement_dates[-1]

    # Calculate event_time
    if event == 'alive':
        event_time_date = last_measurement_date
    else:
        # Death occurs 1-365 days after last measurement
        days_after = random.randint(1, 365)
        event_time_date = last_measurement_date + timedelta(days=days_after)

    # Calculate year and years (time from first measurement to event_time in years)
    time_delta = event_time_date - first_date
    years_value = time_delta.days / 365.25
    year_value = years_value  # Both are the same based on your description

    # Generate baseline values for clinical measurements
    base_creatinine = random.uniform(0.5, 3.0)
    base_hemoglobin = random.uniform(8.0, 16.0)
    base_egfr = random.uniform(15.0, 90.0)
    base_upcr = random.uniform(0.01, 5.0)
    base_acur = random.uniform(0.01, 4.0)

    # Generate rows for each measurement
    rows = []
    for i, measurement_date in enumerate(measurement_dates):
        # Add some variation to clinical measurements over time
        # Simulate disease progression or stability
        progression_factor = i / len(measurement_dates)

        # Calculate age at this measurement
        age_at_measurement = initial_age + ((measurement_date - first_date).days / 365.25)

        # Add temporal variation to clinical values
        creatinine = max(0.1, base_creatinine + random.uniform(-0.3, 0.5) + progression_factor * random.uniform(0, 0.5))
        hemoglobin = max(0.1, base_hemoglobin + random.uniform(-1.0, 1.0) - progression_factor * random.uniform(0, 1.5))
        egfr = max(5.0, base_egfr + random.uniform(-5, 5) - progression_factor * random.uniform(0, 10))
        upcr = max(0.01, base_upcr + random.uniform(-0.5, 0.5) + progression_factor * random.uniform(0, 0.8))
        acur = max(0.01, base_acur + random.uniform(-0.4, 0.4) + progression_factor * random.uniform(0, 0.6))

        row = {
            'subject_id': subject_id,
            'chartdate': measurement_date.strftime('%Y-%m-%d'),
            'event': event,
            'event_time': event_time_date.strftime('%Y-%m-%d'),
            'gender': gender,
            'age_calculated': round(age_at_measurement, 2),
            'Creatinine': round(creatinine, 3),
            'Hemoglobin': round(hemoglobin, 2),
            'eGFR': round(egfr, 2),
            'UPCR': round(upcr, 3),
            'ACUR': round(acur, 3),
            'year': round(year_value, 4),
            'years': round(years_value, 4)
        }
        rows.append(row)

    return rows

def main():
    # Generate data for approximately 100 patients with ~5000 total rows
    num_patients = 100
    target_rows = 5000
    avg_measurements_per_patient = target_rows // num_patients

    all_rows = []

    for patient_id in range(num_patients):
        # Vary the number of measurements per patient (between 30 and 70)
        num_measurements = random.randint(max(30, avg_measurements_per_patient - 20),
                                         avg_measurements_per_patient + 20)

        patient_rows = generate_patient_data(patient_id, num_measurements)
        all_rows.extend(patient_rows)

    # Write to CSV
    fieldnames = ['subject_id', 'chartdate', 'event', 'event_time', 'gender',
                  'age_calculated', 'Creatinine', 'Hemoglobin', 'eGFR', 'UPCR',
                  'ACUR', 'year', 'years']

    output_file = 'dummy_timeseries_data.csv'

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Generated {len(all_rows)} rows of data for {num_patients} patients")
    print(f"Output written to: {output_file}")

    # Print some statistics
    unique_patients = len(set(row['subject_id'] for row in all_rows))
    dead_patients = len(set(row['subject_id'] for row in all_rows if row['event'] == 'dead'))
    alive_patients = unique_patients - dead_patients

    print(f"\nStatistics:")
    print(f"  Total patients: {unique_patients}")
    print(f"  Alive: {alive_patients}")
    print(f"  Dead: {dead_patients}")
    print(f"  Average measurements per patient: {len(all_rows) / unique_patients:.1f}")

if __name__ == '__main__':
    main()
