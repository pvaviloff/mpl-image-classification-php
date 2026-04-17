<?php

ini_set('memory_limit', -1);

require_once 'multilayer_perceptron.php';
require_once 'helper.php';


$labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

$trainData = [];
foreach ($labels as $label) {
    $pathToImages = "./dataset/mnist/training/$label/";
    foreach (getPathToImages($pathToImages) as $i => $imagePath) {
        $imageColors = getImageColor($imagePath);
        $inputs = normalizeInputs($imageColors);
        $trainData[] = [
            'input' => $inputs,
            'number' => $label,
            'label' => encodeLabel($label),
        ];
    }
}
shuffle($trainData);
echo "Training data size: " . count($trainData) . "\n";

$testData = [];
foreach ($labels as $label) {
    $pathToImages = "./dataset/mnist/testing/$label/";
    foreach (getPathToImages($pathToImages) as $imagePath) {
        $imageColors = getImageColor($imagePath);
        $inputs = normalizeInputs($imageColors);
        $testData[] = [
            'input' => $inputs,
            'number' => $label,
        ];
    }
}
echo "Testing data size: " . count($testData) . "\n";

$epochs = 9;
$imageWidth = 28;
$imageHeight = 28;
// 784
$inputSize = $imageWidth * $imageHeight;
$hiddenSize = 64;
// number of numbers
$outputSize = 10;
$learningRate = 0.008;

$perceptron = new MultilayerPerceptron($inputSize, $hiddenSize, $outputSize, $learningRate);

for ($i = 0; $i < $epochs; $i++) {
    echo "Epoch $i\n";
    $correct = 0;
    foreach ($trainData as $train) {
        $perceptron->train($train['input'], $train['label']);

        $predictedNumber = $perceptron->predict($train['input']);
        $actualNumber = $train['number'];
        if ($predictedNumber == $actualNumber) {
            $correct++;
        }
    }
    $trainingAccuracy = $correct / count($trainData) * 100;
    echo "Training accuracy: $trainingAccuracy%\n";

    $correct = 0;
    foreach ($testData as $test) {
        $predictedNumber = $perceptron->predict($test['input']);
        $actualNumber = $test['number'];
        if ($predictedNumber == $actualNumber) {
            $correct++;
        }
    }
    $testingAccuracy = $correct / count($testData) * 100;
    echo "Testing accuracy: $testingAccuracy%\n";
}

$weights = [
    'hiddenWeightsInput' => $perceptron->getHiddenWeightsInput(),
    'hiddenBiases' => $perceptron->getHiddenBiases(),
    'hiddenWeightsOutput' => $perceptron->getHiddenWeightsOutput(),
    'biasesOutput' => $perceptron->getBiasesOutput(),
];

saveWeights($weights);
