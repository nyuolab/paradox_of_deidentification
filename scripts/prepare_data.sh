# usage: source scripts/prepare_data.sh
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
conda activate reid
# create synthetic data
python -m src.data.dummy_data_factory
# tokenize three different kinds of notes
text_cols=("id_text" "deid_text" "condition")
label_cols=("payorfinancialclass" "postal_code_borough" "dyear" "dmonth" "sex" "income_token")
for tokenize_col in ${text_cols[@]}; do
    python -m src.data.pretokenize --model_name bert-base-uncased --input_data_path ./data/dummy_data --basepath ./data --tokenize_col $tokenize_col
    # update classification label
    # for id_text and condition, only update the postal_code borough label
    if [ "$tokenize_col" = "id_text" ] || [ "$tokenize_col" = "condition" ]; then
        label_cols=("postal_code_borough")
    fi
    # go over all labels and update the proper column name
    for label_col in ${label_cols[@]}; do
        python -m src.data.update_classification_label --data_path ./data/dummy_data_tokenized_${tokenize_col} --colname $label_col --output_dir ./data/dummy_data_tokenized_${tokenize_col}_${label_col}
    done
done

