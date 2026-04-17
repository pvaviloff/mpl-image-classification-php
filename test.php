<?php

require_once 'multilayer_perceptron.php';
require_once 'helper.php';


$weights = getWeights();

$imageWidth = 28;
$imageHeight = 28;
$inputSize = $imageWidth * $imageHeight;
$hiddenSize = 64;
$outputSize = 10;

$perceptron = new MultilayerPerceptron($inputSize, $hiddenSize, $outputSize);
$perceptron->setHiddenWeightsInput($weights['hiddenWeightsInput']);
$perceptron->setHiddenBiases($weights['hiddenBiases']);
$perceptron->setHiddenWeightsOutput($weights['hiddenWeightsOutput']);
$perceptron->setBiasesOutput($weights['biasesOutput']);

$labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

$correct = 0;
$count = 0;
foreach ($labels as $label) {
    $pathToImages = "./dataset/mnist/testing/$label/";
    foreach (getPathToImages($pathToImages) as $imagePath) {
        $imageColors = getImageColor($imagePath);
        $inputs = normalizeInputs($imageColors);
        echo "Path to image: $imagePath\n";
        echo "Image number: $label\n";
        $predicted = $perceptron->predict($inputs);
        echo "Predicted number: $predicted\n";

        if ($predicted == $label) {
            $correct++;
        }
        $count++;
    }
}

$testingAccuracy = $correct / $count * 100;
echo "Testing accuracy: $testingAccuracy%\n";
