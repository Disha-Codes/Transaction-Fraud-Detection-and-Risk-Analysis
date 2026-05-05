create database fraud_detect;
use fraud_detect;

rename table cleaned_fraud_data to fraud_data;
select * from fraud_data;

-- fraud by transaction type
select Transaction_Type, count(*) as total_fraud
from fraud_data
where Fraud_Label = 'True'
group by Transaction_Type;

-- high risk users
select * from fraud_data
where Behavior_Risk >= 2;

-- avg risk score
select Fraud_Label, avg(Risk_Score)
from fraud_data
group by Fraud_Label;

