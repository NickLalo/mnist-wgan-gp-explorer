const IMAGE_SIDE = 28;

function parseColor(color) {
  return [1, 3, 5].map(offset => Number.parseInt(color.slice(offset, offset + 2), 16));
}

function imageCanvas(images, imageIndex, paperColor, inkColor) {
  const canvas = document.createElement('canvas');
  canvas.width = IMAGE_SIDE;
  canvas.height = IMAGE_SIDE;
  const context = canvas.getContext('2d');
  const pixels = context.createImageData(IMAGE_SIDE, IMAGE_SIDE);
  const paper = parseColor(paperColor);
  const ink = parseColor(inkColor);
  const offset = imageIndex * IMAGE_SIDE * IMAGE_SIDE;
  for (let index = 0; index < IMAGE_SIDE * IMAGE_SIDE; index += 1) {
    const gray = Math.floor(Math.max(0, Math.min(255, (images[offset + index] + 1) * 127.5)));
    const mix = gray / 255;
    const pixel = index * 4;
    for (let channel = 0; channel < 3; channel += 1) {
      pixels.data[pixel + channel] = Math.round(paper[channel] + (ink[channel] - paper[channel]) * mix);
    }
    pixels.data[pixel + 3] = 255;
  }
  context.putImageData(pixels, 0, 0);
  return canvas;
}

function createCanvas(width, height, paperColor) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  context.fillStyle = paperColor;
  context.fillRect(0, 0, width, height);
  context.imageSmoothingEnabled = false;
  return {canvas, context};
}

function drawImage(context, images, index, x, y, scale, paperColor, inkColor) {
  const source = imageCanvas(images, index, paperColor, inkColor);
  context.drawImage(source, x, y, IMAGE_SIDE * scale, IMAGE_SIDE * scale);
}

async function pngResponse(canvas) {
  const blob = await new Promise((resolve, reject) => {
    canvas.toBlob(value => value ? resolve(value) : reject(new Error('Could not encode image')), 'image/png');
  });
  return new Response(blob, {headers: {'Content-Type': 'image/png'}});
}

export async function renderAllDigits(images, columns, paperColor, inkColor) {
  const rows = 10, scale = 3, gap = 8, tile = IMAGE_SIDE * scale, labelWidth = 42;
  const width = labelWidth + columns * tile + Math.max(columns - 1, 0) * gap;
  const height = rows * tile + (rows - 1) * gap;
  const {canvas, context} = createCanvas(width, height, paperColor);
  for (let index = 0; index < rows * columns; index += 1) {
    const row = Math.floor(index / columns), column = index % columns;
    drawImage(context, images, index, labelWidth + column * (tile + gap), row * (tile + gap), scale, paperColor, inkColor);
  }
  context.fillStyle = inkColor;
  context.font = '18px ui-monospace, monospace';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  for (let row = 0; row < rows; row += 1) context.fillText(String(row), labelWidth / 2, row * (tile + gap) + tile / 2);
  return pngResponse(canvas);
}

export async function renderOneDigit(images, count, scale, paperColor, inkColor) {
  const columns = Math.min(Math.max(1, Math.round(48 / scale)), count);
  const rows = Math.ceil(count / columns);
  const gap = Math.max(0, Math.round(2 * scale));
  const tile = Math.max(1, Math.round(IMAGE_SIDE * scale));
  const renderScale = tile / IMAGE_SIDE;
  const {canvas, context} = createCanvas(
    columns * tile + Math.max(columns - 1, 0) * gap,
    rows * tile + Math.max(rows - 1, 0) * gap,
    paperColor,
  );
  for (let index = 0; index < count; index += 1) {
    drawImage(context, images, index, (index % columns) * (tile + gap), Math.floor(index / columns) * (tile + gap), renderScale, paperColor, inkColor);
  }
  return pngResponse(canvas);
}

export async function renderSingleImage(images, paperColor, inkColor) {
  const scale = 10;
  const {canvas, context} = createCanvas(IMAGE_SIDE * scale, IMAGE_SIDE * scale, paperColor);
  drawImage(context, images, 0, 0, 0, scale, paperColor, inkColor);
  return pngResponse(canvas);
}
