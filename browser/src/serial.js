function abortError() {
  return new DOMException('The operation was aborted.', 'AbortError');
}

export function createSerialExecutor() {
  let tail = Promise.resolve();

  return async function runSerially(task, signal) {
    const previous = tail;
    let release;
    tail = new Promise(resolve => { release = resolve; });

    await previous;
    try {
      if (signal?.aborted) throw abortError();
      return await task();
    } finally {
      release();
    }
  };
}
