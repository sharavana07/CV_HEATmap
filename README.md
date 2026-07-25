# CV_HEATmap

This project trains a convolutional neural network on order-book heatmaps to predict price direction.

## Project structure

- `src/cnn/` — model, training, evaluation, preprocessing, and logging modules
- `src/heatmap/` — heatmap generation utilities
- `outputs/` — checkpoints, plots, results, and logs

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run preprocessing (if needed):
   ```bash
   python src/cnn/preprocess.py
   ```

3. Train the CNN:
   ```bash
   python src/cnn/run_experiment.py
   ```

4. Compare with a simple baseline:
   ```bash
   python src/cnn/logistic_regression.py
   ```

## Notes

- The training pipeline uses temporal splitting to avoid look-ahead bias.
- The model outputs binary labels for `DOWN` vs `UP`.
- Checkpoint and evaluation artifacts are stored in the `outputs/` directory.



code to execute the program :  python -m src.cnn.train --model cnn