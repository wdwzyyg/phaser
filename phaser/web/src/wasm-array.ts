
export let np: null | typeof import("wasm-array") = null;

import("wasm-array")
    .then(mod => np = mod)
    .catch(e => console.error("Error importing wasm-array:", e));