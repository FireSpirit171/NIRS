import re

class Cleaner:

    def clean_duration(self, df):
        pattern = r'^(\d+)( h (\d+)( m)?)?$'
        invalid_count = 0
        valid_rows = []

        for idx, value in enumerate(df['Duration']):
            if re.match(pattern, str(value)):
                valid_rows.append(idx) 
            else:
                invalid_count += 1

        df_cleaned = df.loc[valid_rows]
        print(f"Удалено строк из-за неверного значения Duration: {invalid_count}")
        return df_cleaned