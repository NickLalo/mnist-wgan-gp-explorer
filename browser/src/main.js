import {handleApiRequest} from './engine.js';

try {
  window.__installLocalInference(handleApiRequest);
} catch (error) {
  window.__failLocalInference(error);
  throw error;
}
