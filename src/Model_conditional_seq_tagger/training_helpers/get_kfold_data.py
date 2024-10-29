from sklearn.model_selection import train_test_split

def get_kfold_data(data,random_state=420):

    train_df, val_df =train_test_split(data,test_size=0.2, random_state=random_state)
    train_df=train_df.reset_index(drop=True)
    val_df=val_df.reset_index(drop=True)
    
    return train_df,val_df