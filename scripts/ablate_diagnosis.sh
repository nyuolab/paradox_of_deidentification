# usage: source scripts/ablate_diagnosis.sh
# Trains borough prediction on 3 text types to measure information leakage.
# The deid model is already trained in step 2; this adds id and condition-only.

# Attempt to log in to wandb, and if unsuccessful, prompt for relogin
wandb login || wandb login --relogin

for data in "deid" "id" "condition"; do
    python -m src.attribute_pred.train_bert --config-name bourough_classification_$data.yaml trainer.num_train_epochs=3 data.num_train_samples=100
done
