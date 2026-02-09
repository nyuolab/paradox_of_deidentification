# usage: source scripts/train_predict_attributes.sh

# Attempt to log in to wandb, and if unsuccessful, prompt for relogin
wandb login || wandb login --relogin

for task in "bourough" "dmonth" "dyear" "gender"  "government_pay" "income"; do
    for sample in 100 500; do
        python -m src.attribute_pred.train_bert --config-name ${task}_classification_deid.yaml trainer.num_train_epochs=3 data.num_train_samples=${sample}
    done
done
