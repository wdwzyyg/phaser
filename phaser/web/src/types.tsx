import { NArray } from 'wasm-array';

export type ReconstructionStatus = "queued" | "starting" | "running" | "stopping" | "stopped" | "done";

export interface Reconstruction {
    id: string;
    state: ReconstructionStatus;
    links: Record<string, string>;
};

export interface ReconstructionUpdate {
    msg: 'update';
    id: string;
    state: ReconstructionStatus;
    data: any;
}

export type ProbeData = NArray | null;
export type ObjectData = NArray | null;
export type ProgressData = {
    iters: NArray,
    errors: NArray,
} | null;