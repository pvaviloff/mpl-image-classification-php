<?php

class MultilayerPerceptron
{
    private array $hiddenWeightsInput = [];
    private array $hiddenBiases = [];
    private array $hiddenWeightsOutput = [];
    private array $biasesOutput = [];
    private array $hiddenSums = [];
    private array $hiddenActivations = [];

    public function __construct(
        private int $inputSize,
        private int $hiddenSize,
        private int $outputSize,
        private float $learningRate = 0.01,
    ) {
        $randomizer = new \Random\Randomizer();
        // He initialization for ReLU
        $scale = sqrt(2 / $this->inputSize);
        for ($i = 0; $i < $this->hiddenSize; $i++) {
            for ($j = 0; $j < $this->inputSize; $j++) {
                $this->hiddenWeightsInput[$i][$j] = $randomizer->nextFloat() * $scale;
            }
            $this->hiddenBiases[$i] = $randomizer->nextFloat() * $scale;
        }

        // Xavier initialization for Softmax
        $scale = sqrt(1 / $this->hiddenSize);
        for ($i = 0; $i < $this->outputSize; $i++) {
            for ($j = 0; $j < $this->hiddenSize; $j++) {
                $this->hiddenWeightsOutput[$i][$j] = $randomizer->nextFloat() * $scale;
            }
            $this->biasesOutput[$i] = $randomizer->nextFloat() * $scale;
        }
    }

    private function reluActivation(float $x) : float
    {
        return max(0, $x);
    }

    private function reluDerivate(float $x) : float
    {
        return $x > 0 ? 1 : 0;
    }

    private function softmax(array $outputs) : array
    {
        $maxOutput = max($outputs);
        $expValues = [];
        $sumExpValues = 0;

        foreach ($outputs as $output) {
            $val = exp($output - $maxOutput);
            $expValues[] = $val;
            $sumExpValues += $val;
        }

        $result = [];
        foreach ($expValues as $val) {
            $result[] = $val / $sumExpValues;
        }

        return $result;
    }

    private function forward(array $inputs) : array
    {
        $this->hiddenSums = [];
        foreach ($this->hiddenWeightsInput as $i => $weights) {
            $sum = $this->hiddenBiases[$i];
            foreach ($weights as $j => $weight) {
                $sum += $weight * $inputs[$j];
            }
            $this->hiddenSums[$i] = $sum;
        }

        $this->hiddenActivations = [];
        foreach ($this->hiddenSums as $i => $sum) {
            $this->hiddenActivations[$i] = $this->reluActivation($sum);
        }

        $outputSums = [];
        foreach ($this->hiddenWeightsOutput as $i => $weights) {
            $sum = $this->biasesOutput[$i];
            foreach ($weights as $j => $weight) {
                $sum += $weight * $this->hiddenActivations[$j];
            }
            $outputSums[$i] = $sum;
        }

        return $this->softmax($outputSums);
    }

    private function backward(array $inputs, array $targets, array $outputProbabilities) : void
    {
        $outputDeltas = [];
        foreach ($outputProbabilities as $i => $probability) {
            $outputDeltas[] = $probability - $targets[$i];
        }

        $hiddenDeltas = [];
        foreach ($this->hiddenSums as $i => $sum) {
            $error = 0;
            foreach ($outputDeltas as $j => $outputDelta) {
                $error += $outputDelta * $this->hiddenWeightsOutput[$j][$i];
            }
            $hiddenDeltas[] = $error * $this->reluDerivate($sum);
        }

        foreach ($this->hiddenWeightsOutput as $i => $weights) {
            foreach ($weights as $j => $weight) {
                $this->hiddenWeightsOutput[$i][$j] = $weight - $this->learningRate * $outputDeltas[$i] * $this->hiddenActivations[$j];
            }
        }

        foreach ($this->biasesOutput as $i => $biasOutput) {
            $this->biasesOutput[$i] = $biasOutput - $this->learningRate * $outputDeltas[$i];
        }

        foreach ($this->hiddenWeightsInput as $i => $weights) {
            foreach ($weights as $j => $weight) {
                $this->hiddenWeightsInput[$i][$j] = $weight - $this->learningRate * $hiddenDeltas[$i] * $inputs[$j];
            }
        }

        foreach ($this->hiddenBiases as $i => $hiddenBias) {
            $this->hiddenBiases[$i] = $hiddenBias - $this->learningRate * $hiddenDeltas[$i];
        }
    }

    public function train(array $inputs, array $targets) : void
    {
        $outputProbabilities = $this->forward($inputs);
        $this->backward($inputs, $targets, $outputProbabilities);
    }

    // it’s the exact same algorithm used in forward
    public function predict(array $inputs) : int
    {
        $hiddenSums = [];
        foreach ($this->hiddenWeightsInput as $i => $weights) {
            $sum = $this->hiddenBiases[$i];
            foreach ($weights as $j => $weight) {
                $sum += $weight * $inputs[$j];
            }
            $hiddenSums[$i] = $sum;
        }

        $hiddenActivations = [];
        foreach ($hiddenSums as $i => $sum) {
            $hiddenActivations[$i] = $this->reluActivation($sum);
        }

        $outputSums = [];
        foreach ($this->hiddenWeightsOutput as $i => $weights) {
            $sum = $this->biasesOutput[$i];
            foreach ($weights as $j => $weight) {
                $sum += $weight * $hiddenActivations[$j];
            }
            $outputSums[$i] = $sum;
        }
        $probabilities = $this->softmax($outputSums);

        $maxPredictedNumber = 0;
        $maxProbability = $probabilities[0];
        foreach ($probabilities as $predictedNumber => $probability) {
            if ($probability > $maxProbability) {
                $maxPredictedNumber = $predictedNumber;
                $maxProbability = $probability;
            }
        }

        return $maxPredictedNumber;
    }

    public function getHiddenWeightsInput(): array
    {
        return $this->hiddenWeightsInput;
    }

    public function setHiddenWeightsInput(array $hiddenWeightsInput): void
    {
        $this->hiddenWeightsInput = $hiddenWeightsInput;
    }

    public function getHiddenBiases(): array
    {
        return $this->hiddenBiases;
    }

    public function setHiddenBiases(array $hiddenBiases): void
    {
        $this->hiddenBiases = $hiddenBiases;
    }

    public function getHiddenWeightsOutput(): array
    {
        return $this->hiddenWeightsOutput;
    }

    public function setHiddenWeightsOutput(array $hiddenWeightsOutput): void
    {
        $this->hiddenWeightsOutput = $hiddenWeightsOutput;
    }

    public function getBiasesOutput(): array
    {
        return $this->biasesOutput;
    }

    public function setBiasesOutput(array $biasesOutput): void
    {
        $this->biasesOutput = $biasesOutput;
    }

}
