import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const VIEWS = [
  ["front","正面","面向摄像头，双脚自然分开，双臂离开身体约 15–25°。"],
  ["right","右侧","右侧身体朝向摄像头，手臂不要遮住腰腹轮廓。"],
  ["back","背面","背对摄像头，肩部自然，不要耸肩。"],
  ["left","左侧","左侧身体朝向摄像头，保持自然站姿。"]
].map(([key,name,instruction])=>({key,name,instruction}));
const LM={LS:11,RS:12,LE:13,RE:14,LW:15,RW:16,LH:23,RH:24,LK:25,RK:26,LA:27,RA:28};
const $=id=>document.getElementById(id);
const el={
  modelStatus:$("modelStatus"),heightCm:$("heightCm"),subjectId:$("subjectId"),cameraMode:$("cameraMode"),countdownSeconds:$("countdownSeconds"),
  startCameraBtn:$("startCameraBtn"),autoScanBtn:$("autoScanBtn"),captureBtn:$("captureBtn"),resetBtn:$("resetBtn"),computeBtn:$("computeBtn"),
  downloadJsonBtn:$("downloadJsonBtn"),downloadCsvBtn:$("downloadCsvBtn"),cameraStage:$("cameraStage"),camera:$("camera"),overlay:$("overlay"),
  scanCountdown:$("scanCountdown"),liveHint:$("liveHint"),qualityBox:$("qualityBox"),currentViewName:$("currentViewName"),currentViewInstruction:$("currentViewInstruction"),
  viewStrip:$("viewStrip"),measurementGrid:$("measurementGrid"),threeContainer:$("threeContainer")
};
const state={pose:null,stream:null,raf:0,lastDetect:0,index:0,views:{},measurements:null,auto:false,three:null};

initTiles(); updateGuide(); loadModel();
el.startCameraBtn.addEventListener("click",startCamera);
el.autoScanBtn.addEventListener("click",autoScan);
el.captureBtn.addEventListener("click",()=>capture(true));
el.resetBtn.addEventListener("click",reset);
el.computeBtn.addEventListener("click",compute);
el.downloadJsonBtn.addEventListener("click",downloadJson);
el.downloadCsvBtn.addEventListener("click",downloadCsv);
el.cameraMode.addEventListener("change",()=>{if(state.stream){el.startCameraBtn.textContent="切换摄像头"; el.liveHint.textContent="摄像头模式已更改，请点“切换摄像头”。";}});
window.addEventListener("beforeunload",stopCamera);

async function loadModel(){
  status("正在加载人体模型…","loading");
  try{
    const vision=await FilesetResolver.forVisionTasks(WASM);
    const base={modelAssetPath:MODEL,delegate:"GPU"};
    try{
      state.pose=await PoseLandmarker.createFromOptions(vision,{baseOptions:base,runningMode:"VIDEO",numPoses:1,minPoseDetectionConfidence:.5,minPosePresenceConfidence:.5,minTrackingConfidence:.5,outputSegmentationMasks:true});
    }catch(e){
      console.warn("GPU unavailable, using CPU",e);
      state.pose=await PoseLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:MODEL},runningMode:"VIDEO",numPoses:1,minPoseDetectionConfidence:.5,minPosePresenceConfidence:.5,minTrackingConfidence:.5,outputSegmentationMasks:true});
    }
    status("人体模型已就绪","ready");
  }catch(e){console.error(e); status("模型加载失败","error"); el.liveHint.textContent="模型加载失败，请检查网络后刷新。";}
}
function status(t,c){el.modelStatus.textContent=t;el.modelStatus.className=`status-pill ${c}`;}

async function startCamera(){
  if(!state.pose){el.liveHint.textContent="人体模型还在加载，请稍等。";return;}
  if(!navigator.mediaDevices?.getUserMedia){el.liveHint.textContent="当前浏览器无法调用摄像头。请用 iPhone Safari 并通过 HTTPS 打开。";return;}
  stopCamera();
  const facing=el.cameraMode.value;
  try{
    let stream;
    try{stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{exact:facing},width:{ideal:1920},height:{ideal:1080},frameRate:{ideal:30,max:30}}});}
    catch{stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:facing},width:{ideal:1920},height:{ideal:1080},frameRate:{ideal:30,max:30}}});}
    state.stream=stream; el.camera.srcObject=stream; el.camera.setAttribute("playsinline",""); await el.camera.play();
    el.cameraStage.classList.toggle("mirrored",facing==="user");
    el.captureBtn.disabled=false; el.autoScanBtn.disabled=false; el.startCameraBtn.textContent=facing==="user"?"前置摄像头已开启":"后置摄像头已开启";
    el.liveHint.textContent="把完整身体放进画面：头顶、手腕和脚底都不要出框。"; speak("摄像头已经开启。请把完整身体放进画面。",true); loop();
  }catch(e){console.error(e); el.liveHint.textContent=`摄像头开启失败：${cameraError(e)}`;}
}
function stopCamera(){if(state.raf)cancelAnimationFrame(state.raf);state.raf=0;if(state.stream){state.stream.getTracks().forEach(t=>t.stop());state.stream=null;}}
function loop(){
  const now=performance.now();
  if(el.camera.readyState>=2&&state.pose&&now-state.lastDetect>150){state.lastDetect=now;try{const r=state.pose.detectForVideo(el.camera,now);drawPose(r);guideLive(r);r.close?.();}catch(e){console.warn(e);}}
  state.raf=requestAnimationFrame(loop);
}
function resizeOverlay(){if(!el.camera.videoWidth)return;el.overlay.width=el.camera.videoWidth;el.overlay.height=el.camera.videoHeight;}
function drawPose(r){
  resizeOverlay();const c=el.overlay.getContext("2d");c.clearRect(0,0,el.overlay.width,el.overlay.height);const p=r.landmarks?.[0];if(!p)return;
  const links=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
  c.strokeStyle="rgba(125,211,252,.95)";c.fillStyle="white";c.lineWidth=Math.max(2,el.overlay.width/500);
  for(const[a,b]of links){if((p[a].visibility??1)<.35||(p[b].visibility??1)<.35)continue;c.beginPath();c.moveTo(p[a].x*el.overlay.width,p[a].y*el.overlay.height);c.lineTo(p[b].x*el.overlay.width,p[b].y*el.overlay.height);c.stroke();}
  for(const q of p){if((q.visibility??1)<.4)continue;c.beginPath();c.arc(q.x*el.overlay.width,q.y*el.overlay.height,Math.max(2.5,el.overlay.width/420),0,Math.PI*2);c.fill();}
}
function guideLive(r){
  const p=r.landmarks?.[0];if(!p){el.liveHint.textContent="没有检测到人体，请后退并保证全身入镜。";return;}
  const pts=[p[11],p[12],p[15],p[16],p[27],p[28]];const ok=pts.every(q=>(q.visibility??0)>.35&&q.x>.015&&q.x<.985&&q.y>.015&&q.y<.985);
  el.liveHint.textContent=ok?`可以拍摄：${VIEWS[state.index].name}`:"请调整距离：肩、手腕和脚踝都要完整入镜。";
}

async function autoScan(){
  if(state.auto)return; if(!state.stream)await startCamera(); if(!state.stream)return;
  state.auto=true; el.autoScanBtn.disabled=true; el.captureBtn.disabled=true;
  try{
    for(let i=0;i<VIEWS.length;i++){
      state.index=i;updateGuide();const v=VIEWS[i];speak(`准备拍摄${v.name}。${v.instruction}`);await sleep(900);await countdown(Number(el.countdownSeconds.value)||7);
      const ok=await capture(false);if(!ok){speak(`${v.name}没有识别成功，请重新站好。`);await sleep(800);i--;continue;}
      speak(`${v.name}完成。`);await sleep(900);
    }
    state.index=0;updateGuide(); if(state.views.front&&state.views.right){el.computeBtn.disabled=false;compute();}
    speak("四个视角扫描完成。",true);
  }finally{state.auto=false;el.autoScanBtn.disabled=false;el.captureBtn.disabled=false;}
}
async function countdown(sec){el.scanCountdown.classList.remove("hidden");for(let n=sec;n>=1;n--){el.scanCountdown.textContent=n;if(n<=3)speak(String(n),true);await sleep(1000);}el.scanCountdown.textContent="保持";await sleep(350);el.scanCountdown.classList.add("hidden");}

async function capture(advance=true){
  if(!state.pose||!state.stream||el.camera.readyState<2)return false;
  const v=VIEWS[state.index];let r;
  try{
    r=state.pose.detectForVideo(el.camera,performance.now());const p=r.landmarks?.[0],maskObj=r.segmentationMasks?.[0];if(!p||!maskObj){quality("未取得完整人体，请重新站好再拍。",false);return false;}
    const mask=new Float32Array(maskObj.getAsFloat32Array());const {w,h}=inferMaskSize(maskObj,mask.length,el.camera.videoWidth/el.camera.videoHeight);
    const bbox=maskBounds(mask,w,h,.5);if(!bbox||bbox.height<h*.45){quality("人体在画面中太小或不完整，请调整距离。",false);return false;}
    const landmarks=p.map(q=>({x:q.x,y:q.y,z:q.z,visibility:q.visibility??0,presence:q.presence??0}));
    const canvas=document.createElement("canvas");canvas.width=el.camera.videoWidth;canvas.height=el.camera.videoHeight;canvas.getContext("2d").drawImage(el.camera,0,0,canvas.width,canvas.height);
    const qScore=captureQuality(landmarks,bbox,w,h);
    state.views[v.key]={key:v.key,name:v.name,image:canvas.toDataURL("image/jpeg",.72),mask,maskW:w,maskH:h,landmarks,bbox,quality:qScore,capturedAt:new Date().toISOString()};
    quality(`${v.name}已采集 · 质量 ${qScore}/100`,qScore>=60);renderTiles();
    if(advance){state.index=(state.index+1)%VIEWS.length;updateGuide();}
    el.computeBtn.disabled=!(state.views.front&&state.views.right);return true;
  }catch(e){console.error(e);quality(`采集失败：${e.message||e}`,false);return false;}finally{r?.close?.();}
}
function inferMaskSize(obj,len,aspect){let w=Number(obj.width)||0,h=Number(obj.height)||0;if(w&&h&&w*h===len)return{w,h};h=Math.max(1,Math.round(Math.sqrt(len/aspect)));w=Math.max(1,Math.round(len/h));if(w*h!==len){w=Math.max(1,Math.round(Math.sqrt(len*aspect)));h=Math.max(1,Math.round(len/w));}return{w,h};}
function maskBounds(mask,w,h,t=.5){let minX=w,minY=h,maxX=-1,maxY=-1;for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(mask[y*w+x]>=t){if(x<minX)minX=x;if(x>maxX)maxX=x;if(y<minY)minY=y;if(y>maxY)maxY=y;}return maxX<0?null:{minX,minY,maxX,maxY,width:maxX-minX+1,height:maxY-minY+1};}
function captureQuality(p,b,w,h){let s=100;const needed=[11,12,15,16,23,24,27,28];const avg=needed.reduce((a,i)=>a+(p[i]?.visibility??0),0)/needed.length;if(avg<.7)s-=Math.round((.7-avg)*80);if(b.minY<2||b.maxY>h-3)s-=18;if(b.minX<2||b.maxX>w-3)s-=10;if(b.height<h*.55)s-=20;return Math.max(0,Math.min(100,s));}
function quality(t,good){el.qualityBox.textContent=t;el.qualityBox.className=`quality-box ${good?"good":"bad"}`;}

function compute(){
  const f=state.views.front,s=state.views.right;if(!f||!s)return;const height=Number(el.heightCm.value);if(!(height>=120&&height<=230)){alert("请输入正确身高（120–230cm）。");return;}
  const fg=geometry(f,height),sg=geometry(s,height);if(!fg||!sg){quality("无法从轮廓计算尺寸，请重新扫描正面和右侧。",false);return;}
  const chestF=torsoWidth(f,fg,.29),waistF=torsoWidth(f,fg,.70),hipF=torsoWidth(f,fg,1.02);
  const chestS=torsoWidth(s,sg,.29),waistS=torsoWidth(s,sg,.70),hipS=torsoWidth(s,sg,1.02);
  if([chestF,waistF,hipF,chestS,waistS,hipS].some(v=>!Number.isFinite(v))){quality("部分身体截面没有识别到，请重新扫描。",false);return;}
  const p=f.landmarks,scale=fg.cmPerPx;
  const shoulder=distPx(p[LM.LS],p[LM.RS],f.maskW,f.maskH)*scale;
  const leftSleeve=(distPx(p[LM.LS],p[LM.LE],f.maskW,f.maskH)+distPx(p[LM.LE],p[LM.LW],f.maskW,f.maskH))*scale;
  const rightSleeve=(distPx(p[LM.RS],p[LM.RE],f.maskW,f.maskH)+distPx(p[LM.RE],p[LM.RW],f.maskW,f.maskH))*scale;
  const shoulderDiff=Math.abs((p[LM.LS].y-p[LM.RS].y)*f.maskH)*scale;
  const sideP=s.landmarks;const shoulderX=((sideP[LM.LS].x+sideP[LM.RS].x)/2)*s.maskW;const hipX=((sideP[LM.LH].x+sideP[LM.RH].x)/2)*s.maskW;
  const m={version:"0.1-mobile",subjectId:el.subjectId.value||"TEST",measuredAt:new Date().toISOString(),inputHeightCm:height,
    chestCircumferenceCm:r1(ellipse(chestF,chestS)),waistCircumferenceCm:r1(ellipse(waistF,waistS)),hipCircumferenceCm:r1(ellipse(hipF,hipS)),
    shoulderWidthCm:r1(shoulder),sleeveLengthLeftCm:r1(leftSleeve),sleeveLengthRightCm:r1(rightSleeve),shoulderHeightDifferenceCm:r1(shoulderDiff),
    frontChestWidthCm:r1(chestF),sideChestDepthCm:r1(chestS),frontWaistWidthCm:r1(waistF),sideWaistDepthCm:r1(waistS),frontHipWidthCm:r1(hipF),sideHipDepthCm:r1(hipS),
    torsoHorizontalOffsetSideCm:r1((shoulderX-hipX)*sg.cmPerPx),captureQualityFront:f.view.quality,captureQualityRight:s.view.quality,
    note:"V0.1 engineering estimate — not production measurements"
  };
  state.measurements=m;renderMeasurements(m);render3D(m);el.downloadJsonBtn.disabled=false;el.downloadCsvBtn.disabled=false;quality("尺寸已计算。下一步请和版师手工量体对照。",true);
}
function geometry(view,height){const p=view.landmarks,b=view.bbox;if(!b)return null;const cmPerPx=height/b.height;const sy=((p[11].y+p[12].y)/2),hy=((p[23].y+p[24].y)/2),kx=((p[25].y+p[26].y)/2);const cx=((p[11].x+p[12].x+p[23].x+p[24].x)/4);return{view,cmPerPx,shoulderY:sy,hipY:hy,kneeY:kx,centerX:cx};}
function torsoWidth(view,g,fraction){const yNorm=fraction<=1?g.shoulderY+(g.hipY-g.shoulderY)*fraction:g.hipY+(g.kneeY-g.hipY)*(fraction-1)*.35;const px=rowSegmentAverage(view,yNorm,g.centerX);return px?px*g.cmPerPx:NaN;}
function rowSegmentAverage(view,yNorm,cxNorm){const vals=[];for(let d=-2;d<=2;d++){const y=Math.max(0,Math.min(view.maskH-1,Math.round(yNorm*(view.maskH-1))+d));const seg=segment(view.mask,view.maskW,view.maskH,y,cxNorm,.5);if(seg)vals.push(seg.width);}if(!vals.length)return null;vals.sort((a,b)=>a-b);return vals[Math.floor(vals.length/2)];}
function segment(mask,w,h,y,cxNorm,t){const cx=Math.round(cxNorm*(w-1));const segs=[];let st=-1;for(let x=0;x<w;x++){const inside=mask[y*w+x]>=t;if(inside&&st<0)st=x;if((!inside||x===w-1)&&st>=0){const en=inside&&x===w-1?x:x-1;segs.push({left:st,right:en,width:en-st+1,mid:(st+en)/2});st=-1;}}if(!segs.length)return null;return segs.find(a=>cx>=a.left&&cx<=a.right)||segs.reduce((a,b)=>Math.abs(b.mid-cx)<Math.abs(a.mid-cx)?b:a,segs[0]);}
function distPx(a,b,w,h){return Math.hypot((a.x-b.x)*w,(a.y-b.y)*h);}
function ellipse(d1,d2){const a=Math.max(d1,.1)/2,b=Math.max(d2,.1)/2;return Math.PI*(3*(a+b)-Math.sqrt((3*a+b)*(a+3*b)));}

function renderMeasurements(m){const rows=[["胸围（估算）",m.chestCircumferenceCm],["腰围（估算）",m.waistCircumferenceCm],["臀围（估算）",m.hipCircumferenceCm],["肩宽",m.shoulderWidthCm],["左袖长",m.sleeveLengthLeftCm],["右袖长",m.sleeveLengthRightCm],["左右肩高度差",m.shoulderHeightDifferenceCm],["胸部正面宽度",m.frontChestWidthCm],["胸部侧面厚度",m.sideChestDepthCm],["腰部正面宽度",m.frontWaistWidthCm],["腰部侧面厚度",m.sideWaistDepthCm],["臀部正面宽度",m.frontHipWidthCm],["臀部侧面厚度",m.sideHipDepthCm],["侧面肩-胯偏移",m.torsoHorizontalOffsetSideCm]];el.measurementGrid.className="measurement-grid";el.measurementGrid.innerHTML=rows.map(([n,v])=>`<div class="metric"><span>${n}</span><strong>${v??"—"}</strong><small>cm</small></div>`).join("");}

function render3D(m){clear3D();const c=el.threeContainer;c.innerHTML="";const scene=new THREE.Scene(),cam=new THREE.PerspectiveCamera(34,c.clientWidth/c.clientHeight,.1,100);cam.position.set(0,4.5,12);const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));renderer.setSize(c.clientWidth,c.clientHeight);c.appendChild(renderer.domElement);const controls=new OrbitControls(cam,renderer.domElement);controls.enableDamping=true;controls.target.set(0,3.6,0);scene.add(new THREE.HemisphereLight(0xffffff,0x223344,2.4));const dl=new THREE.DirectionalLight(0xffffff,2);dl.position.set(4,7,5);scene.add(dl);
  const mat=new THREE.MeshStandardMaterial({roughness:.75,metalness:.02});const scale=.055;const rings=[[7.0,m.shoulderWidthCm*.95,m.sideChestDepthCm*.75],[6.3,m.frontChestWidthCm,m.sideChestDepthCm],[4.7,m.frontWaistWidthCm,m.sideWaistDepthCm],[3.5,m.frontHipWidthCm,m.sideHipDepthCm],[2.9,m.frontHipWidthCm*.86,m.sideHipDepthCm*.84]];scene.add(torsoMesh(rings,scale,mat));const head=new THREE.Mesh(new THREE.SphereGeometry(.47,28,18),mat);head.scale.set(.8,1.08,.88);head.position.set(0,8.1,0);scene.add(head);const neck=new THREE.Mesh(new THREE.CylinderGeometry(.29,.32,.55,20),mat);neck.position.set(0,7.55,0);scene.add(neck);const grid=new THREE.GridHelper(10,10,0x334155,0x1e293b);scene.add(grid);
  let active=true;const resize=()=>{if(!c.isConnected)return;cam.aspect=c.clientWidth/c.clientHeight;cam.updateProjectionMatrix();renderer.setSize(c.clientWidth,c.clientHeight);};window.addEventListener("resize",resize);(function anim(){if(!active)return;controls.update();renderer.render(scene,cam);requestAnimationFrame(anim);})();state.three={dispose(){active=false;window.removeEventListener("resize",resize);controls.dispose();renderer.dispose();scene.traverse(o=>{o.geometry?.dispose?.();o.material?.dispose?.();});}};
}
function torsoMesh(rings,scale,mat){const n=56,v=[],idx=[];rings.forEach(([y,w,d])=>{for(let i=0;i<n;i++){const a=i/n*Math.PI*2;v.push(Math.cos(a)*(w/2)*scale,y,Math.sin(a)*(d/2)*scale);}});for(let r=0;r<rings.length-1;r++)for(let i=0;i<n;i++){const j=(i+1)%n,a=r*n+i,b=r*n+j,c=(r+1)*n+i,d=(r+1)*n+j;idx.push(a,c,b,b,c,d);}const g=new THREE.BufferGeometry();g.setAttribute("position",new THREE.Float32BufferAttribute(v,3));g.setIndex(idx);g.computeVertexNormals();return new THREE.Mesh(g,mat);}
function clear3D(){state.three?.dispose?.();state.three=null;el.threeContainer.innerHTML='<div class="empty-3d">计算尺寸后生成 3D 模型</div>';}

function initTiles(){el.viewStrip.innerHTML=VIEWS.map(v=>`<div class="view-tile" data-view="${v.key}"><div class="meta"><strong>${v.name}</strong><br>未采集</div></div>`).join("");}
function renderTiles(){for(const v of VIEWS){const box=el.viewStrip.querySelector(`[data-view="${v.key}"]`),d=state.views[v.key];if(d){box.classList.add("captured");box.innerHTML=`<img src="${d.image}" alt="${v.name}"><div class="meta"><strong>${v.name}</strong><br>质量 ${d.quality}/100</div>`;}else{box.classList.remove("captured");box.innerHTML=`<div class="meta"><strong>${v.name}</strong><br>未采集</div>`;}}}
function updateGuide(){const v=VIEWS[state.index];el.currentViewName.textContent=v.name;el.currentViewInstruction.textContent=v.instruction;}
function reset(){state.index=0;state.views={};state.measurements=null;state.auto=false;updateGuide();initTiles();quality("尚未采集",true);el.computeBtn.disabled=true;el.downloadJsonBtn.disabled=true;el.downloadCsvBtn.disabled=true;el.measurementGrid.className="measurement-grid empty-state";el.measurementGrid.textContent="完成至少“正面 + 右侧”后即可计算。";clear3D();}

function downloadJson(){if(!state.measurements)return;const payload={...state.measurements,capturedViews:VIEWS.filter(v=>state.views[v.key]).map(v=>({key:v.key,quality:state.views[v.key].quality,capturedAt:state.views[v.key].capturedAt}))};download(`${safe(payload.subjectId)}-body-scan-v0.1.json`,JSON.stringify(payload,null,2),"application/json");}
function downloadCsv(){if(!state.measurements)return;const m=state.measurements;const h=["subject_id","scan_date","height_cm","scan_chest_cm","manual_chest_cm","scan_waist_cm","manual_waist_cm","scan_hip_cm","manual_hip_cm","scan_shoulder_width_cm","manual_shoulder_width_cm","scan_left_sleeve_cm","manual_left_sleeve_cm","scan_right_sleeve_cm","manual_right_sleeve_cm","scan_shoulder_height_diff_cm","manual_shoulder_height_diff_cm","front_quality","right_quality","notes"];const r=[m.subjectId,m.measuredAt,m.inputHeightCm,m.chestCircumferenceCm,"",m.waistCircumferenceCm,"",m.hipCircumferenceCm,"",m.shoulderWidthCm,"",m.sleeveLengthLeftCm,"",m.sleeveLengthRightCm,"",m.shoulderHeightDifferenceCm,"",m.captureQualityFront,m.captureQualityRight,""];download(`${safe(m.subjectId)}-validation.csv`,[h,r].map(x=>x.map(csv).join(",")).join("\n"),"text/csv;charset=utf-8");}
function download(name,text,type){const u=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement("a");a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
function csv(v){const s=String(v??"");return /[",\n]/.test(s)?`"${s.replaceAll('"','""')}"`:s;}function safe(s){return String(s||"scan").replace(/[^a-z0-9-_]+/gi,"-");}
function cameraError(e){if(e?.name==="NotAllowedError")return"没有摄像头权限，请在 Safari 网站设置中允许摄像头。";if(e?.name==="NotFoundError")return"没有找到可用摄像头。";if(e?.name==="NotReadableError")return"摄像头正被其他应用占用。";return e?.message||String(e);}
function speak(text,interrupt=false){try{if(!("speechSynthesis" in window))return;if(interrupt)speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang="zh-CN";u.rate=.95;speechSynthesis.speak(u);}catch{}}
function sleep(ms){return new Promise(r=>setTimeout(r,ms));}function r1(v){return Number.isFinite(v)?Math.round(v*10)/10:null;}
