<?php

function getImageColor(string $path): array
{
    $image = imagecreatefrompng($path);
    $width = imagesx($image);
    $height = imagesy($image);

    $colors = [];
    for ($x = 0; $x < $width; $x++) {
        for ($y = 0; $y < $height; $y++) {
            $color = imagecolorat($image, $x, $y);
            $colors[$x][$y] = $color;
        }
    }
    imagedestroy($image);

    return $colors;
}

function normalizeInputs(array $imageColors) : array
{
    $inputs = [];
    foreach ($imageColors as $colorData) {
        foreach ($colorData as $color) {
            $filteredColor = $color > 20 ? $color : 0;
            $inputs[] = $filteredColor / 255;
        }
    }

    return $inputs;
}

function getPathToImages(string $path) : array
{
    $imagesPath = [];
    foreach (scandir($path) as $imageFileName) {
        if (is_dir($imageFileName)) continue;
        $imagesPath[] = $path . $imageFileName;
    }

    return $imagesPath;
}

function encodeLabel(int $label) : array
{
    $labelInput = [];
    for ($i = 0; $i < 10; $i++) {
        $labelInput[] = ($label == $i) ? 1 : 0;
    }

    return $labelInput;
}

function saveWeights(array $weights) : void
{
    $file = new SplFileObject('./model/weights', 'w+');
    $file->fwrite(serialize($weights));
}

function getWeights() : array
{
    $file = new SplFileObject('./model/weights', 'r');
    $file->rewind();
    return unserialize($file->fgets());
}
