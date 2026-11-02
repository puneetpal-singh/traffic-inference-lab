import fs from 'node:fs/promises';import path from 'node:path';import {fileURLToPath} from 'node:url';import {createHash} from 'node:crypto';
const root=path.dirname(fileURLToPath(import.meta.url));const manifest=JSON.parse(await fs.readFile(path.join(root,'asset-manifest.json'),'utf8'));const hash=b=>createHash('sha256').update(b).digest('hex');
const requested=process.argv[2];if(requested&&!manifest.files.some(a=>a.path===requested))throw new Error('Unknown asset path; use a path from asset-manifest.json');
for(const [i,a] of manifest.files.entries()){
  if(requested&&requested!==a.path)continue;
  const dest=path.resolve(root,a.path);if(!dest.startsWith(root+path.sep))throw new Error('Unsafe asset path');
  try{const current=await fs.readFile(dest);if(hash(current)===a.sha256){console.log(`[${i+1}/${manifest.files.length}] Verified ${a.path}`);continue;}}catch(e){if(e.code!=='ENOENT')throw e;}
  const url=new URL(a.path.replace(/^dist\//,''),manifest.origin+'/');console.log(`[${i+1}/${manifest.files.length}] Downloading ${a.path} (${(a.bytes/1048576).toFixed(1)} MiB)…`);
  const r=await fetch(url,{signal:AbortSignal.timeout(180000)});if(!r.ok)throw new Error(`HTTP ${r.status}: ${a.path}`);const b=Buffer.from(await r.arrayBuffer());if(b.length!==a.bytes||hash(b)!==a.sha256)throw new Error(`Integrity mismatch: ${a.path}`);await fs.mkdir(path.dirname(dest),{recursive:true});await fs.writeFile(dest,b);console.log(`  SHA-256 verified`);
}
console.log('Assets ready. Run: node server.mjs');
