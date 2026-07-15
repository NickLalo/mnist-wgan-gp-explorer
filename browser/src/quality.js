export function candidateCount(requested, oversample = 1.2) {
  return Math.max(requested, requested + 4, Math.ceil(requested * oversample));
}

function stableOrder(values, higherIsBetter = true) {
  return Array.from(values, (_, index) => index).sort((left, right) => {
    const difference = higherIsBetter ? values[right] - values[left] : values[left] - values[right];
    return difference || left - right;
  });
}

function qualityRanks(values, higherIsBetter = true) {
  const order = stableOrder(values, higherIsBetter);
  const ranks = new Float32Array(values.length);
  order.forEach((localIndex, rank) => {
    ranks[localIndex] = values.length === 1 ? 1 : 1 - rank / (values.length - 1);
  });
  return ranks;
}

function lowerMedian(values) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor((sorted.length - 1) / 2)];
}

function slice1d(values, indices) {
  return indices.map(index => values[index]);
}

export function selectQualityIndices({
  labels,
  keepPerClass,
  criticScores,
  logits,
  unsupported,
  disconnected,
  profiles,
  rejectionThreshold = 0.15,
  detachedInkThreshold = 0.1,
  unsupportedInkThreshold = 0.03,
}) {
  const digits = [...new Set(labels)].sort((left, right) => left - right);
  const selected = [];

  for (const digit of digits) {
    const indices = Array.from(labels, (_, index) => index).filter(index => labels[index] === digit);
    if (indices.length < keepPerClass) throw new Error(`digit ${digit} has too few candidates`);
    const scores = new Float32Array(indices.length);
    const criticRanks = qualityRanks(slice1d(criticScores, indices));

    const margins = indices.map(index => {
      const offset = index * 10;
      let alternative = -Infinity;
      for (let candidate = 0; candidate < 10; candidate += 1) {
        if (candidate !== digit) alternative = Math.max(alternative, logits[offset + candidate]);
      }
      return logits[offset + digit] - alternative;
    });
    const marginRanks = qualityRanks(margins);

    const medians = new Float32Array(4);
    for (let feature = 0; feature < 4; feature += 1) {
      medians[feature] = lowerMedian(indices.map(index => profiles[index * 4 + feature]));
    }
    const deviations = indices.map(index => {
      const values = new Float32Array(4);
      for (let feature = 0; feature < 4; feature += 1) {
        values[feature] = Math.abs(profiles[index * 4 + feature] - medians[feature]);
      }
      return values;
    });
    const scales = new Float32Array(4);
    for (let feature = 0; feature < 4; feature += 1) {
      scales[feature] = Math.max(lowerMedian(deviations.map(row => row[feature])), 1e-3);
    }
    const outliers = deviations.map(row => (
      row.reduce((total, value, feature) => total + value / scales[feature], 0) / 4
    ));
    const profileRanks = qualityRanks(outliers, false);
    const unsupportedRanks = qualityRanks(slice1d(unsupported, indices), false);
    const disconnectedRanks = qualityRanks(slice1d(disconnected, indices), false);

    for (let local = 0; local < indices.length; local += 1) {
      scores[local] = 0.30 * criticRanks[local]
        + 0.45 * marginRanks[local]
        + 0.10 * profileRanks[local]
        + 0.075 * unsupportedRanks[local]
        + 0.075 * disconnectedRanks[local];
    }

    const rejected = [];
    for (let local = 0; local < keepPerClass; local += 1) {
      const index = indices[local];
      let predicted = 0;
      for (let candidate = 1; candidate < 10; candidate += 1) {
        if (logits[index * 10 + candidate] > logits[index * 10 + predicted]) predicted = candidate;
      }
      const artifact = disconnected[index] > detachedInkThreshold
        && unsupported[index] > unsupportedInkThreshold;
      if (scores[local] < rejectionThreshold || predicted !== digit || artifact) rejected.push(local);
    }

    const replaceCount = Math.min(rejected.length, indices.length - keepPerClass);
    const rejectedSet = new Set(
      rejected.sort((left, right) => scores[left] - scores[right] || left - right).slice(0, replaceCount),
    );
    for (let local = 0; local < keepPerClass; local += 1) {
      if (!rejectedSet.has(local)) selected.push(indices[local]);
    }
    stableOrder(Array.from(scores.slice(keepPerClass)))
      .slice(0, replaceCount)
      .forEach(local => selected.push(indices[keepPerClass + local]));
  }
  return selected;
}
