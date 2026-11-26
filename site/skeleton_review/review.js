'use strict';
const $=id=>document.getElementById(id);
const video=$('vid'), FPS=25, LAST=13494, block=250, stride=4+3*jointNames.length;
const canvases=[$('original'),$('overlay')], ctx=canvases.map(c=>c.getContext('2d'));
const players=['רון בן ישי','מאור האס'], ids=['Ron_Ben_Ishay','Maor_Haas'];
const labels={foot:'רגל',chest:'חזה',head:'ראש',shoulder:'כתף',knee:'ברך',shin:'שוק',unknown:'לא ידוע'};
const allCases=window.touchCases;
const presets=[[7572,1,34,210,'foot'],[7640,1,34,0,'ankles'],[3659,0,22,101,'foot'],[250,1,3,0,'absent']];
let frame=7572,who=1,ticket=0,currentLabelsKey='',storageOK=true, pendingSeek=null;
const storeKey='graz2022-95616a9-sapiens-review-v1';
let marks={};
try { marks=JSON.parse(localStorage.getItem(storeKey)||'{}')||{}; } catch {storageOK=false;}
const approvedInReview = [
 {player:'Ron_Ben_Ishay', frame:8977, rally:39},
 {player:'Ron_Ben_Ishay', frame:13376, rally:59},
 {player:'Maor_Haas', frame:9892, rally:43},
];
const approvalBatchKey = storeKey + '-approved-8977-13376-9892';
function applyReviewApprovals(){
 if(storageOK && localStorage.getItem(approvalBatchKey))return;
 for(const item of approvedInReview){
  const key=item.player+'|'+item.frame;
  marks[key]={...marks[key], status:'ok'};
 }
 if(storageOK){
  localStorage.setItem(storeKey, JSON.stringify(marks));
  localStorage.setItem(approvalBatchKey,'1');
 }
}
try { applyReviewApprovals(); } catch {storageOK=false;}
const jointIndex=new Map(jointNames.map((n,i)=>[n,i]));
const skeleton=[['nose','neck'],['nose','left_eye','left_ear'],['nose','right_eye','right_ear'],['left_shoulder','right_shoulder'],['left_hip','right_hip']];
for(const side of ['left','right']){
 const names=items=>items.map(s=>side+'_'+s);
 skeleton.push(['neck',side+'_shoulder'],names(['shoulder','elbow','wrist']),names(['shoulder','hip','knee','ankle','heel','big_toe','small_toe','heel']),names(['ankle','big_toe']));
}
const chains=skeleton.map(path=>path.map(n=>jointIndex.get(n)));
const body=[...new Set(chains.flat())];
const cache=new Map(), waiting=new Map();
window.receivePose=async function(index,b64){
 const wait=waiting.get(index);if(!wait)return;
 try{
  const bytes=Uint8Array.from(atob(b64),s=>s.charCodeAt(0));
  const data=new Int16Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer());
  const n=Math.min(block,LAST+1-index*block);
  if(data.length!==n*2*stride)throw Error('נתוני הפריימים אינם שלמים');
  cache.set(index,data);if(cache.size>7){const outside=[...cache.keys()].filter(i=>Math.abs(i-Math.floor(frame/block))>2);if(outside.length)cache.delete(outside[0]);}
  wait.resolve(data);
 }catch(e){wait.reject(e);}finally{waiting.delete(index);}
};
function getChunk(i){
 if(cache.has(i))return Promise.resolve(cache.get(i));
 if(waiting.has(i))return waiting.get(i).promise;
 let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});waiting.set(i,{promise,resolve,reject});
 const sc=document.createElement('script');sc.src='data/pose-'+i+'.js';sc.onload=()=>sc.remove();
 sc.onerror=()=>{sc.remove();waiting.delete(i);reject(Error('קובץ השלד לא נטען'));};document.head.appendChild(sc);return promise;
}
function timeString(f){const time=f/FPS;return Math.floor(time/60)+':'+(time%60).toFixed(2).padStart(5,'0');}
const owns=(obj,key)=>Object.prototype.hasOwnProperty.call(obj,key);
function key(){return ids[who]+'|'+frame;}
function save(){
 try{localStorage.setItem(storeKey,JSON.stringify(marks));storageOK=true;}catch{storageOK=false;}
 $('msg').textContent=storageOK?'':'השמירה המקומית אינה זמינה. אפשר לייצא את ההחלטות כעת.';
}
function refreshMarks(){
 const k=key(),m=marks[k]||{};
 if(currentLabelsKey!==k){$('comment').value=m.note||'';currentLabelsKey=k;}
 $('ok').classList.toggle('on',m.status==='ok');$('manual').classList.toggle('on',m.status==='manual');
 const n=Object.values(marks).filter(x=>x.status).length;
 $('saved').textContent=n?'החלטות שסומנו: '+n:'';
}
function freezeFrame(){if(!video.paused){video.pause();jump(frame);}}
function decision(value){freezeFrame(); const k=key();marks[k]={...marks[k],status:value,note:$('comment').value};save();refreshMarks();}
$('ok').onclick=()=>decision('ok');$('manual').onclick=()=>decision('manual');
$('clear').onclick=()=>{freezeFrame();delete marks[key()];currentLabelsKey='';save();refreshMarks();};
$('comment').onfocus=freezeFrame;
$('comment').oninput=()=>{const k=key(),note=$('comment').value;
 if(!note&&!(marks[k]&&marks[k].status)){delete marks[k];}else{marks[k]={...marks[k],note};}save();
};
function updateUI(){
 $('name').textContent=players[who];
 const match=allCases.find(a=>a[0]===frame&&a[1]===who);
 $('where').textContent=(match?'ראלי '+match[2]+' · נגיעה '+match[3]+' · ':'')+'פריים '+frame+' · '+timeString(frame);
 $('frame').value=frame;$('slider').value=frame;
 $('legend').textContent=$('color').checked?'כתום: ציון מתחת ל־0.50 · תכלת: 0.50 ומעלה':'כל ציוני הביטחון מוצגים באותו צבע';
 const id=String(frame)+'-'+who;if($('cases').value!==id)$('cases').value=owns(options,id)?id:'';
 refreshMarks();
}
function aspect(src){
 let [x,y,w,h]=src; const aspect=4/3;
 if(w/h<aspect)w=h*aspect;else h=w/aspect;
 const scale=Math.min(1,1920/w,1080/h);w*=scale;h*=scale;
 return [Math.max(0,Math.min(1920-w,x-w/2)),Math.max(0,Math.min(1080-h,y-h/2)),w,h];
}
function paint(data,flag){
 const offset=((frame%block)*2+who)*stride;
 const present=data&&data[offset]!==-32768;
 let crop=[0,0,1920,1080];
 if(present&&!$('wide').checked){
  const [x1,y1,x2,y2]=Array.from(data.subarray(offset,offset+4),k=>k/4);
  crop=aspect([(x1+x2)/2,(y1+y2)/2,Math.max(x2-x1,75)*1.48,Math.max(y2-y1,90)*1.34]);
 }
 const [x,y,w,h]=crop,W=960,H=720,scale=Math.min(W/w,H/h),dw=w*scale,dh=h*scale,ox=(W-dw)/2,oy=(H-dh)/2;
 for(const c of ctx){c.fillStyle='#16251f';c.fillRect(0,0,W,H);if(video.readyState>=2)c.drawImage(video,x,y,w,h,ox,oy,dw,dh);}
 const c=ctx[1];
 if(!present){c.save();c.font='bold 25px Arial';c.direction='rtl';c.textAlign='center';c.fillStyle='#102921';c.fillRect(0,H-50,W,50);c.fillStyle='white';c.fillText(flag||'אין הצעת שלד לשחקן בפריים הזה',W/2,H-16);c.restore();return;}
 function point(i){
  const n=offset+4+3*i;const px=data[n],py=data[n+1],s=data[n+2];
  if(px===-32768||py===-32768)return null;
  return [(px/4-x)*scale+ox,(py/4-y)*scale+oy,s===-32768?null:s/1000];
 }
 const full=$('wide').checked, radius=full?2.7:5, color=$('color').checked;
 const bodyPositions=body.map(i=>[i,point(i)]);const low=(a,b)=>color&&(a[2]===null||b[2]===null||Math.min(a[2],b[2])<.5);
 c.lineCap='round';c.lineJoin='round';
 for(const pass of [0,1])for(const chain of chains)for(let k=1;k<chain.length;k++){
  const a=point(chain[k-1]),b=point(chain[k]);if(!a||!b)continue;
  c.strokeStyle=pass===0?'#10222c':(low(a,b)?'#ffac44':'#3cfff0');c.lineWidth=pass===0?(full?4:6):(full?1.8:3.4);c.beginPath();c.moveTo(a[0],a[1]);c.lineTo(b[0],b[1]);c.stroke();
 }
 if($('dense').checked){const extras=new Set(body);for(let i=0;i<jointNames.length;i++){if(extras.has(i))continue;const a=point(i);if(!a)continue;c.fillStyle=low(a,a)?'#ffa537':'#f4fcff';c.beginPath();c.arc(a[0],a[1],full?1:1.8,0,2*Math.PI);c.fill();}}
 for(const [,a] of bodyPositions){if(!a)continue;c.beginPath();c.arc(a[0],a[1],radius,0,2*Math.PI);c.fillStyle=low(a,a)?'#ffac44':'#63fff0';c.fill();c.lineWidth=full?1.15:1.8;c.strokeStyle='#122630';c.stroke();}
}
function draw(){
 const serial=++ticket,i=Math.floor(frame/block);updateUI();if(video.readyState<2||video.seeking)return;
 if(cache.has(i)){paint(cache.get(i));return;}
 paint(null,'טוען את נתוני השלד…');
 getChunk(i).then(data=>{if(ticket===serial&&!video.seeking)paint(data);}).catch(e=>{$('err').textContent='לא ניתן להציג את השלד. '+e.message;});
}
function jump(v){const n=Number(v);if(!Number.isFinite(n))return;frame=Math.max(0,Math.min(LAST,Math.round(n)));pendingSeek=frame;video.pause();updateUI();if(video.readyState<1)return;
 const goal=(frame+.25)/FPS;if(!video.seeking&&Math.abs(video.currentTime-goal)<.0001){pendingSeek=null;draw();}else video.currentTime=goal;
}
const options={};
function optionGroup(title,entries,featured=false){
 const g=document.createElement('optgroup');g.label=title;
 for(const a of entries){const id=a[0]+'-'+a[1];if(owns(options,id))continue;options[id]=true;const opt=document.createElement('option');opt.value=id;
 opt.textContent=players[a[1]]+' · ראלי '+a[2]+' · '+timeString(a[0])+(a[3]?' · '+(labels[a[4]]||a[4]):(a[4]==='ankles'?' · רצף קרסוליים':' · פער בשלד'));
 g.appendChild(opt);} $('cases').appendChild(g);
}
const empty=document.createElement('option');empty.value='';empty.textContent='בחירת מהלך';$('cases').appendChild(empty);
optionGroup('דוגמאות מהבדיקה',presets);optionGroup('נגיעות כף רגל שנבחרו לבדיקה',allCases.filter(a=>a[7]));optionGroup('יתר הנגיעות המתועדות',allCases);
$('cases').onchange=()=>{if(!$('cases').value)return;const [f,p]=$('cases').value.split('-').map(Number);who=p;$('who').value=p;jump(f);};
$('who').onchange=()=>{who=Number($('who').value);draw();};
for(const id of ['wide','dense','color'])$(id).onchange=draw;
$('prev').onclick=()=>jump(frame-1);$('next').onclick=()=>jump(frame+1);$('go').onclick=()=>jump($('frame').value);
$('frame').onkeydown=e=>{if(e.key==='Enter')jump($('frame').value);};$('slider').oninput=()=>jump($('slider').value);
$('speed').onchange=()=>video.playbackRate=Number($('speed').value);video.playbackRate=.5;
$('play').onclick=()=>{if(!video.paused)video.pause();else video.play().catch(e=>{$('err').textContent=e.message});};
video.onpause=()=>{$('play').textContent='ניגון';};video.onplay=()=>{$('play').textContent='עצירה';};
video.onloadedmetadata=()=>jump(pendingSeek??7572);
video.onseeked=()=>{const f=Math.min(LAST,Math.floor(video.currentTime*FPS+.0001));if(pendingSeek!==null&&f!==pendingSeek)return;frame=f;pendingSeek=null;draw();};
video.onloadeddata=draw;
video.onerror=()=>{$('err').textContent='לא ניתן לפתוח את סרטון המשחק. יש לפתוח את התצוגה מתיקיית המשחק בדפדפן התומך ב־MP4.';};
function onVideoFrame(now,meta){
 if(!video.paused&&!video.seeking&&pendingSeek===null){frame=Math.min(LAST,Math.floor(meta.mediaTime*FPS+.001));draw();if(frame%block>222&&frame+block-frame%block<=LAST)getChunk(Math.floor(frame/block)+1).catch(()=>{});}
 video.requestVideoFrameCallback(onVideoFrame);
}
if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(onVideoFrame);else video.ontimeupdate=()=>{if(!video.paused&&!video.seeking&&pendingSeek===null){frame=Math.min(LAST,Math.floor(video.currentTime*FPS));draw();}};
$('export').onclick=()=>{
 const rows=[['player','frame_v3','seconds_v3','touch_id','rally','decision','note']];
 for(const [k,v] of Object.entries(marks)){const [id,f]=k.split('|');const a=allCases.find(z=>z[0]===+f&&ids[z[1]]===id);
 const reviewed=approvedInReview.find(item=>item.frame===+f&&item.player===id);
 rows.push([id,+f,+f/FPS,a?.[3]??'',a?.[2]??reviewed?.rally??'',v.status==='ok'?'תקין':v.status==='manual'?'צריך תיוג ידני':'',v.note||'']);}
 const esc=v=>{let s=String(v);if(/^[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
 const blob=new Blob(['\uFEFF'+rows.map(r=>r.map(esc).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});
 const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='graz-pose-decisions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);
};
getChunk(30).catch(()=>{});updateUI();if(video.readyState>=1)jump(7572);
