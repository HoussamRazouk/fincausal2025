import pandas as pd

#Some data inconsistency which have been fixed manually  line 102 line 602 line 
data=pd.read_csv("training_data_en.csv",sep=';')

data.columns

data['is the answer part of the text']=data.apply(lambda row: row['Answer'] not in row['Text'],axis=1)
print(f'The number of answers which are not part of the text {sum(data['is the answer part of the text'])}')


inconstant_data_indexes= data[data['is the answer part of the text']].index.values

for idx in inconstant_data_indexes:
    
    print (f"""{data.iloc[idx]['ID']};{data.iloc[idx]['Text']};{data.iloc[idx]['Question']};{data.iloc[idx]['Answer']}""")

#note

#3373;Keller has improved its processes for the capture and recording of environmental incidents and, as a consequence, we have seen a slight increase in the number of reported environmental incidents in 2017 (12% year on year).;What caused the slight increase in the number of reported environmental incidents in 2017?;Life on land Keller has improved its processes for the capture and recording of environmental incidents
## missing   Life on land 

#5364.3;Profit after tax rose from $68.1m in 2016 to $117.6m, driven by underlying earnings growth and a one-off tax credit as result of changes to US tax legislation.;What factors contributed to the rise in profit after tax from $68.1m in 2016 to $117.6m?;underlying earnings growth and a one off tax credit
## missing - in one off

#4039.a;Copper sales volumes reflected within revenue increased from 634,100 tonnes in 2016 to 657,700 tonnes in 2017 increasing revenue by $122.0 million. This increase was mainly due to Antucoya which achieved commercial production on 1 April 2016, and which recorded sales volumes of 54,900 tonnes reflected within revenue from that point onwards.;What led to the increase in copper sales volumes from 634,100 tonnes in 2016 to 657,700 tonnes in 2017?;a $122.0 million increase in revenue
## increasing revenue by $122.0 million have been re-written in the answer 

#6014.b;All divisions were profitable in 2017, with Motor and Commercial reporting significant improvements in operating profit compared to 2016 due mainly to the non-repeat of the Ogden discount rate change. This was partially offset by a decrease in Home, primarily due to lower prior-year reserve releases and the impact of higher EoW claims. Rescue operating profit of £43.5 million (2016: £42.8 million) is included in the Rescue and other personal lines result.;What factors account for the partial offset caused by a decrease in Home?;lower prior-year re serve releases and the impact of higher EoW claims
# extra space in re serve

#2564;What effect does the UK tax regime, which facilitates the collection of tax from life insurance policyholders by making an equivalent charge within the corporate tax of the Company, have on the total tax charge for insurance companies?;What are the implications the UK tax regime have on the total tax charge for insurance companies?;comprises both this element and an element more closely related to normal corporation tax
# no text in this example


#2587;• No malus or clawback provisions apply to base salary, because "natural" malus and clawback are built into the structure as described in the Notes to the Policy table. • There is no set maximum mo netary value for the salary of Executive Directors.;What factor lies behind the decision that no malus or clawback provisions apply to base salary?;natural malus and clawback are built into the structure as described in the Notes to the Policy table
#  "" are not included in the answer

#3681.a;The financial results for the year reflect a period of transition for Circassia. Gross profit Gross margin increased from 65% to 78%. This was mainly due to the contribution of revenues from the AstraZeneca collaboration for the full year, which due to the agreement structure have a 100% gross margin.;What factor caused the gross margin to increase from 65% to 78%?;the contribution of revenues from the AstraZeneca collaboration for the full year
#? maybe extra space

#5221;• No malus or clawback provisions apply to base salary, because "natural" malus and clawback are built into the structure as described in the Notes to the Policy table. • There is no set maximum monetary value for the salary of Executive Directors.;What accounts for the absence of malus or clawback provisions applied to base salary?;natural malus and clawback are built into the structure as described in the Notes to the Policy table
# same 2587 as "" are not included in the answer

#5269.3.b;Given current market conditions we believed that it was appropriate to operate with a lower level of gearing and used the proceeds from asset sales to repay these shares in full, rather than re- financing. Consequently, we have also simplified our corporate structure.;What is the reason for simplifying the corporate structure?;Given current market conditions we believed that it was appropriate to operate with a lower level of gearing and used the proceeds from asset sales to repay these shares in full, rather than re-financing
# extra space in re-financing

#3146.b;Investments in unlisted equity securities, by their nature, involve a higher degree of valuation and performance uncertainties and liquidity risks than investments in listed securities and therefore may be more difficult to realise.;What factor explains why investments in unlisted equity securities involve higher degrees of valuation and performance uncertainties and liquidity risks compared to investments in listed securities?;Their nature
# capital T

#3965;Cineworld's current Remuneration Policy (the "Policy") was approved by shareholders at the AGM in 2017 and was based on the profile of Cineworld before the acquisition of Regal. As such, the Committee considered that it was an appropriate time to review the Policy;Why did the Committee considered that it was an appropriate time to review the Policy?;Remuneration Policy Cineworld's current Remuneration Policy (the "Policy") was approved by shareholders at the AGM in 2017 and was based on the profile of Cineworld before the acquisition of Regal
# extra Remuneration Policy 

#4047;Due to the significant reliance on foreign currencies in Georgia's economy, currency-induced credit risk is a significant component of credit risk, which relates to risks arising from foreign currency- denominated loans to unhedged borrowers in the Group's portfolio.;What is the effect of the significant reliance on foreign currencies in Georgia's economy?;currency-induced credit risk is a significant component of credit risk, which relates to risks arising from foreign currency-denominated loans to unhedged borrowers in the Group's portfolio
# extra space currency-denominated



