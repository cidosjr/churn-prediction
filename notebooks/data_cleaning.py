import pickle
import pandas as pd

path = '/home/cid/repos/churn-prediction'

scale_mms_age =                  pickle.load(open(path + '/src/features/scale_mms_age.pkl', 'rb'))
scale_mms_balance =              pickle.load(open(path + '/src/features/scale_mms_balance.pkl', 'rb'))
scale_mms_estimated_salary =     pickle.load(open(path + '/src/features/scale_mms_estimated_salary.pkl', 'rb'))
scale_ss_credit_score =          pickle.load(open(path + '/src/features/scale_ss_credit_score.pkl', 'rb'))

target_encode_age_bins =         pickle.load(open(path + '/src/features/target_encode_age_bins.pkl', 'rb'))
target_encode_gender =           pickle.load(open(path + '/src/features/target_encode_gender.pkl', 'rb'))
target_encode_has_cr_card =      pickle.load(open(path + '/src/features/target_encode_has_cr_card.pkl', 'rb'))
target_encode_is_active_member = pickle.load(open(path + '/src/features/target_encode_is_active_member.pkl', 'rb'))
target_encode_num_of_products =  pickle.load(open(path + '/src/features/target_encode_num_of_products.pkl', 'rb'))
target_encode_tenure =           pickle.load(open(path + '/src/features/target_encode_tenure.pkl', 'rb'))
target_encode_age_group =        pickle.load(open(path + '/src/features/target_encode_age_group.pkl', 'rb'))

def data_cleaning(data):
    data.columns = ['row_number', 'customer_id', 'surname', 'credit_score', 'geography', 'gender', 'age', 'tenure', 'balance', 'num_of_products', 'has_cr_card','is_active_member', 'estimated_salary', 'exited']

    # age_group 
    data.loc[(data['age'] >= 17) & (data['age'] <= 39), 'age_group'] = '18-39'
    data.loc[(data['age'] >= 40) & (data['age'] <= 59), 'age_group'] = '40-59'
    data.loc[data['age'] >= 60, 'age_group'] = '>60'

    # age_bins
    data['age_bins'] = pd.cut(data['age'], 7)
           
        
    # credit_score
    data['credit_score'] = scale_ss_credit_score.fit_transform( data[['credit_score']].values )

    # age
    data['age'] = scale_mms_age.fit_transform( data[['age']].values )

    # balance
    data['balance'] = scale_mms_balance.fit_transform( data[['balance']].values )

    # estimated_salary
    data['estimated_salary'] = scale_mms_estimated_salary.fit_transform( data[['estimated_salary']].values )

    # target encode - gender
    data.loc[:, 'gender'] = data.loc[:, 'gender'].map(target_encode_gender)

    # num_of_products
    data.loc[:, 'num_of_products'] = data.loc[:, 'num_of_products'].map(target_encode_num_of_products)

    # target encode - tenure
    data.loc[:, 'tenure'] = data.loc[:, 'tenure'].map(target_encode_tenure) # 

    # target encode - has_cr_card
    data.loc[:, 'has_cr_card'] = data.loc[:, 'has_cr_card'].map(target_encode_has_cr_card)

    # target encode - is_active_member
    data.loc[:, 'is_active_member'] = data.loc[:, 'is_active_member'].map(target_encode_is_active_member)

    # target encode - age_bins
    data.loc[:, 'age_bins'] = data.loc[:, 'age_bins'].map(target_encode_age_bins).astype('float64')
    
    # target encode - age_group
    data.loc[:, 'age_group'] = data.loc[:, 'age_group'].map(target_encode_age_group)
    
    selected_cols =['credit_score',
                    'age_bins',
                    'gender',
                    'age',
                    'num_of_products',
                    'age_group',
                    'is_active_member',
                    'balance',
                    'has_cr_card',
                    'tenure',
                    'estimated_salary']

    return data[selected_cols]
