// @ts-ignore
const ctx: Worker = self as any;

onmessage = async (event) => {
    ctx.postMessage(`[WORKER_TS] ping`)
}
