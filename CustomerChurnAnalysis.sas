/* STEP 1: Import CSV data into SAS workspace */
/* Reads customer churn dataset and stores it as a SAS table named churn_data */
proc import datafile="/home/u63821302/sasuser.v94/customer_churn.csv"
    out=churn_data
    dbms=csv
    replace;
    guessingrows=max; /* Allows SAS to read all rows to infer column types accurately */
run;

/* STEP 2: Explore dataset structure */
/* Shows variable names, types, and attributes */
proc contents data=churn_data; run;

/* Summarizes numeric columns and counts missing values */
proc means data=churn_data n nmiss; run;

/* STEP 3: Handle Missing Values using Multiple Imputation */
/* Applies PROC MI to impute missing values in numeric variables */
proc mi data=churn_data out=churn_imputed nimpute=1 seed=12345;
    var tenure MonthlyCharges TotalCharges;
run;

/* STEP 4: Convert TotalCharges from character to numeric */
/* Sometimes TotalCharges is read as character due to missing or invalid values */
data churn_imputed;
    set churn_imputed;
    TotalCharges_num = input(TotalCharges, best32.); /* Convert to numeric */
    drop TotalCharges;
    rename TotalCharges_num = TotalCharges; /* Replace old column with numeric version */
run;

/* STEP 5: Create binary churn_flag */
/* Converts 'Yes'/'No' churn labels into numeric binary (1 = churned, 0 = not churned) */
data churn_imputed;
    set churn_imputed;
    if Churn = "Yes" then churn_flag = 1;
    else churn_flag = 0;
run;

/* STEP 6: Feature Engineering */
/* Create buckets for tenure and MonthlyCharges to capture non-linear trends */
data churn_imputed;
    set churn_imputed;
    length tenure_group $10 MonthlyCharges_group $10;

    /* Grouping tenure into 4 categories */
    if tenure < 12 then tenure_group = "0-12 mo";
    else if tenure < 24 then tenure_group = "12-24 mo";
    else if tenure < 48 then tenure_group = "24-48 mo";
    else tenure_group = "48+ mo";

    /* Grouping MonthlyCharges into 3 categories */
    if MonthlyCharges < 35 then MonthlyCharges_group = "Low";
    else if MonthlyCharges < 70 then MonthlyCharges_group = "Medium";
    else MonthlyCharges_group = "High";
run;

/* STEP 7: Check for multicollinearity among numeric predictors */
/* VIF > 10 suggests high multicollinearity */
proc reg data=churn_imputed;
    model churn_flag = tenure MonthlyCharges TotalCharges / vif;
run;

/* STEP 8: Logistic Regression to predict churn */
/* Includes categorical and numeric variables */
/* 'descending' tells SAS to model probability of churn_flag=1 */
proc logistic data=churn_imputed descending;
    class gender Partner Dependents PhoneService MultipleLines InternetService 
          Contract PaymentMethod / param=ref; /* Reference coding for dummy vars */
    model churn_flag = gender Partner Dependents PhoneService MultipleLines InternetService 
                       Contract PaymentMethod tenure MonthlyCharges TotalCharges SeniorCitizen;
run;

/* STEP 9: Generate predicted probabilities from the model */
proc logistic data=churn_imputed descending;
    class gender Partner Dependents PhoneService MultipleLines InternetService 
          Contract PaymentMethod / param=ref;
    model churn_flag = gender Partner Dependents PhoneService MultipleLines InternetService 
                       Contract PaymentMethod tenure MonthlyCharges TotalCharges SeniorCitizen;
    output out=logit_out p=predicted_prob; /* Save probabilities in new dataset */
run;

/* STEP 10: Evaluate model using confusion matrix and classification metrics */
/* Convert predicted probability to binary class using 0.5 threshold */
data eval;
    set logit_out;
    predicted_class = (predicted_prob >= 0.5);
run;

/* Generate confusion matrix: actual vs predicted */
proc freq data=eval;
    tables churn_flag*predicted_class / out=conf_matrix nocol norow nopercent;
run;

/* Manually calculate Precision, Recall, and F1 Score from confusion matrix */
data f1_metrics;
    set conf_matrix end=last;
    retain TP FP FN TN 0;
    
    /* Identify TP, FP, FN, TN based on combinations */
    if churn_flag=1 and predicted_class=1 then TP=Count;
    else if churn_flag=0 and predicted_class=1 then FP=Count;
    else if churn_flag=1 and predicted_class=0 then FN=Count;
    else if churn_flag=0 and predicted_class=0 then TN=Count;

    /* Calculate metrics after last row */
    if last then do;
        precision = TP / (TP + FP);
        recall = TP / (TP + FN);
        f1_score = 2 * (precision * recall) / (precision + recall);
        output;
    end;
run;

proc print data=f1_metrics; run;

/* STEP 11: Plot ROC Curve to assess model performance */
proc logistic data=churn_imputed descending plots=roc;
    class gender Partner Dependents PhoneService MultipleLines InternetService 
          Contract PaymentMethod / param=ref;
    model churn_flag = gender Partner Dependents PhoneService MultipleLines InternetService 
                       Contract PaymentMethod tenure MonthlyCharges TotalCharges SeniorCitizen;
run;

/* STEP 12: Manual K-Fold Cross-Validation (5-fold) */
/* STEP 12.1: Assign each row to a fold */
%let k = 5;

data kfold_data;
    set churn_imputed;
    fold_id = mod(_N_, &k) + 1; /* Random assignment from 1 to k */
run;

/* STEP 12.2: Define macro for repeated training/testing */
%macro kfold_logit(k=5);
    %do i = 1 %to &k;
        /* Split into train and test sets for fold i */
        data train test;
            set kfold_data;
            if fold_id = &i then output test;
            else output train;
        run;

        /* Train logistic model on training set */
        proc logistic data=train descending outmodel=logit_model;
            class gender Partner Dependents PhoneService MultipleLines InternetService 
                  Contract PaymentMethod / param=ref;
            model churn_flag = gender Partner Dependents PhoneService MultipleLines InternetService 
                               Contract PaymentMethod tenure MonthlyCharges TotalCharges SeniorCitizen;
        run;

        /* Score test set using trained model */
        proc logistic inmodel=logit_model;
            score data=test out=score&i(rename=(P_1=predicted_prob)); /* Save predicted probabilities */
        run;
    %end;
%mend;

%kfold_logit(k=5);

/* STEP 12.3: Combine results from all folds and evaluate */
/* Append all scored data */
data all_scores;
    set score1-score5;
    predicted_class = (predicted_prob >= 0.5);
run;

/* Confusion matrix across all folds */
proc freq data=all_scores;
    tables churn_flag*predicted_class / norow nocol nopercent;
    title "5-Fold Cross-Validation Confusion Matrix";
run;
