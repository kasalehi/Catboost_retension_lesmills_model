from sqlalchemy import create_engine, text
import pandas as pd
from urllib.parse import quote_plus
# run this once before your script
import os; 
os.makedirs('monthlychurned', exist_ok=True)

server   = r'LMNZLREPORT01\LM_RPT'                 # named instance
database = 'LMNZ_Report'
driver   = 'ODBC Driver 17 for SQL Server'         # use 18 if installed

username = 'LMNZ_ReportUser'
password = 'LMNZ_ReportUser'                           # avoid @ : / ? in raw form

# If password contains special chars, do: password = quote_plus(password)
engine = create_engine(
    f"mssql+pyodbc://{username}:{quote_plus(password)}@{server}/{database}"
    f"?driver={driver.replace(' ', '+')}&Encrypt=yes&TrustServerCertificate=yes"
)

listItem=["2024-11-15", "2023-07-23", "2025-02-02"]
for i in range(len(listItem)):
    try:

        query = f"""


        DECLARE @TestDate DATE = '{listItem[i]}';
        DECLARE @ChurnWindowEnd DATE = DATEADD(WEEK, 52, @TestDate);
        DECLARE @MinVisits12m INT = 12;
        DECLARE @MinVisitsWks INT = 6;

        WITH base_members AS (
            SELECT
                fam.MemberID,
                fam.MembershipID,
                CAST(fam.[ActiveOn Date] AS date) AS active_on_date,
                CAST(fam.[End Date]      AS date) AS end_date
            FROM fact.LMNZ_ALLMemberships AS fam
            WHERE fam.[ActiveOn Date] <= @TestDate
            AND (fam.[End Date] >= @TestDate OR fam.[End Date] IS NULL)
            AND fam.SubCategory NOT IN ('Unvaccinated')
            AND fam.Category IN ('Contract')
        ),
        labels AS (
            SELECT
                bm.MembershipID,
                CASE
                    WHEN bm.end_date IS NOT NULL
                    AND bm.end_date >  @TestDate
                    AND bm.end_date <  @ChurnWindowEnd
                    THEN 1 ELSE 0
                END AS churn_flag,
                CASE
                    WHEN bm.end_date IS NULL OR bm.end_date <= @TestDate THEN NULL
                    WHEN bm.end_date <= DATEADD(MONTH,  1, @TestDate) THEN  1
                    WHEN bm.end_date <= DATEADD(MONTH,  2, @TestDate) THEN  2
                    WHEN bm.end_date <= DATEADD(MONTH,  3, @TestDate) THEN  3
                    WHEN bm.end_date <= DATEADD(MONTH,  4, @TestDate) THEN  4
                    WHEN bm.end_date <= DATEADD(MONTH,  5, @TestDate) THEN  5
                    WHEN bm.end_date <= DATEADD(MONTH,  6, @TestDate) THEN  6
                    WHEN bm.end_date <= DATEADD(MONTH,  7, @TestDate) THEN  7
                    WHEN bm.end_date <= DATEADD(MONTH,  8, @TestDate) THEN  8
                    WHEN bm.end_date <= DATEADD(MONTH,  9, @TestDate) THEN  9
                    WHEN bm.end_date <= DATEADD(MONTH, 10, @TestDate) THEN 10
                    WHEN bm.end_date <= DATEADD(MONTH, 11, @TestDate) THEN 11
                    WHEN bm.end_date <= DATEADD(MONTH, 12, @TestDate) THEN 12
                    ELSE NULL
                END AS end_month_bucket
            FROM base_members bm
        ),
        mw AS (
            SELECT
                a.MembershipID,
                CAST(a.WeekBeginningDate AS date) AS week,
                COALESCE(a.WeekVisits, 0) AS engagement,
                a.[OnPauseThisWeek] AS pause
            FROM repo.MemberWeeklyAttendanceCounts a
            INNER JOIN base_members bm
                ON a.MembershipID = bm.MembershipID
            WHERE a.WeekBeginningDate >= DATEADD(YEAR, -1, @TestDate)
            AND a.WeekBeginningDate  <  @TestDate
        ),
        eligible AS (
            SELECT
                mw.MembershipID
            FROM mw
            GROUP BY mw.MembershipID
            HAVING SUM(mw.engagement) >= @MinVisits12m
            AND COUNT(DISTINCT CASE WHEN mw.engagement > 0 THEN mw.week END) >= @MinVisitsWks
        )
        SELECT
            mw.MembershipID AS member_id,
            mw.week,
            mw.engagement,
            CAST(mw.pause AS int) AS paused,
            lbl.churn_flag,
            lbl.end_month_bucket
        FROM mw
        JOIN eligible e
        ON e.MembershipID = mw.MembershipID
        LEFT JOIN labels lbl
        ON lbl.MembershipID = mw.MembershipID
        ORDER BY member_id, week;
        """

        df = pd.read_sql_query(query, engine)

        df.to_csv(f'monthlychurned/churned_{listItem[i]}.csv')
    except:
        print("something not correct")