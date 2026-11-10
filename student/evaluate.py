"""评估最终A/B返回值，保存带时间戳的结果，并展示已有运行的排行榜。"""
import argparse
import datetime as dt
import html
import json
import os
from pathlib import Path
import webbrowser
from urllib.parse import quote
from _runner import CONTRACT, ROOT, read_jsonl, sha, stamp

STYLE = """body{font:17px/1.6 'Microsoft YaHei',sans-serif;color:#243247;background:#f4f6fa;margin:0}
header{background:#2654a6;color:white;padding:22px 32px}h1{margin:0;font-size:29px}
main{max-width:1200px;margin:24px auto;padding:0 25px}section,article{background:white;padding:22px;margin:18px 0}
h2{font-size:23px;color:#2654a6;margin:0 0 10px}table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:12px;border-bottom:1px solid #dde3ed}th{color:#2654a6}a{color:#2654a6}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fb;padding:16px;font:15px/1.6 Consolas,monospace}
button,select{font:inherit;padding:5px 10px}nav{display:flex;gap:18px;align-items:center;flex-wrap:wrap}
.bad{color:#ae2c2c}.muted{color:#657389}.cols{display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:700px){.cols{grid-template-columns:1fr}}"""

DETAIL = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Pipeline运行记录</title><style>__STYLE__</style>
<header><h1>Pipeline运行记录</h1><div id="name"></div></header>
<main><section id="stats"></section><nav>
<select id="question"></select><label><input type="checkbox" id="wrong">只看错误题</label>
<button id="prev">上一题</button><button id="next">下一题</button></nav>
<article><h2 id="verdict"></h2><p id="error"></p><div class="cols">
<div><h3 id="at"></h3><p id="aa"></p></div><div><h3 id="bt"></h3><p id="ba"></p></div></div></article>
<article><h2>完整调用过程</h2><div id="trace"></div></article></main>
<script>
const D=__DATA__;const $=x=>document.getElementById(x);let rows=[];
$('name').textContent=D.name+' · '+D.config.model+' (thinking)'+(D.config.mock?' · MOCK模拟，非正式成绩':'');
$('stats').textContent=D.summary;
$('wrong').disabled=!D.metrics;
function render(){const r=rows[$('question').selectedIndex];if(!r)return;
$('verdict').textContent=r.id+' · 函数返回 '+(r.prediction||'无效/缺失')+(r.answer?' · 标签 '+r.answer+' · '+(r.correct?'正确':'错误'):'');
$('error').textContent=r.error||'';for(const k of ['a','b']){$(k+'t').textContent='论文'+k.toUpperCase()+'：'+r['paper_'+k].title;$(k+'a').textContent=r['paper_'+k].abstract;}
$('trace').replaceChildren();for(const call of r.calls){const block=document.createElement('details');block.open=true;
const title=document.createElement('summary');title.textContent='调用 '+call.local_index+' · '+(call.response?call.response.finish_reason:call.error||'状态未确认');block.append(title);
for(const [name,value] of [['消息',JSON.stringify(call.messages,null,2)],['模型最终文本',call.response?.content||''],['模型思考',call.response?.reasoning||''],['用量',JSON.stringify(call.response?.usage||{})]]){const label=document.createElement('h3');label.textContent=name;const pre=document.createElement('pre');pre.textContent=value;block.append(label,pre);}
$('trace').append(block);}if(!r.calls.length)$('trace').textContent='此题没有模型调用记录。';}
function fill(){rows=D.rows.filter(r=>!$('wrong').checked||r.correct===false);$('question').replaceChildren();for(const r of rows){const o=document.createElement('option');o.textContent=r.id; $('question').append(o);}document.querySelectorAll('article').forEach(el=>el.hidden=!rows.length);render();}
$('wrong').onchange=fill;$('question').onchange=render;$('prev').onclick=()=>{$('question').selectedIndex=Math.max(0,$('question').selectedIndex-1);render();};$('next').onclick=()=>{$('question').selectedIndex=Math.min(rows.length-1,$('question').selectedIndex+1);render();};fill();
</script>"""

def unique(rows, name):
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(name+'中存在重复题号')
    return {r['id']:r for r in rows}

def evaluate(predictions, labels):
    predictions = Path(predictions)
    folder = predictions.parent
    config = json.loads((folder/'config.json').read_text(encoding='utf8'))
    if config.get('contract') != CONTRACT:
        raise ValueError('旧框架结果不能与新pipeline结果混用，请重新运行')
    gold = unique(json.loads(Path(labels).read_text(encoding='utf8')), '标签')
    data = unique(read_jsonl(folder/'dataset.jsonl'), '数据')
    if not gold or set(gold) != set(data) or any(g['answer'] not in ('A','B') for g in gold.values()):
        raise ValueError('标签须与本次完整测试集逐题对应，且只能是A/B')
    rows = unique(read_jsonl(predictions), '结果')
    if set(rows)-set(gold):
        raise ValueError('结果包含未知题号')
    events = read_jsonl(folder/'calls.jsonl')
    reserved = [r for r in events if r.get('event')=='reserved']
    if len({r['call_id'] for r in reserved}) != len(reserved) or len(reserved)>config['budget_limit']:
        raise ValueError('调用账本重复或超预算')
    details = []
    for rid,g in gold.items():
        row = rows.get(rid,{})
        value = row.get('prediction')
        if type(value) is not str or value not in ('A','B'):
            value = None
        details.append({'id':rid,'prediction':value,'answer':g['answer'],'correct':value==g['answer']})
    completed = [r for r in events if r.get('event')=='completed']
    usage = {k:sum(r['response'].get('usage',{}).get(k,0) or 0 for r in completed) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    correct = sum(r['correct'] for r in details)
    summary_file = folder/'summary.json'
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    return {'n':len(gold),'submitted':len(rows),'missing':len(gold)-len(rows),'correct':correct,
            'accuracy':correct/len(gold),'invalid':sum(r['prediction'] is None for r in details),
            'model_errors':sum(e.get('event')=='error' for e in events),
            'length_finishes':sum(e['response'].get('finish_reason')=='length' for e in completed),
            'calls':len(reserved),'budget_limit':config['budget_limit'],'usage':usage,
            'external_calls':0 if config['mock'] else len(reserved),'mock':config['mock'],
            'wall_seconds':summary.get('wall_seconds'),'labels_sha256':sha(Path(labels).read_bytes()),
            'details':details}

def save_detail(folder, metrics, token):
    config=json.loads((folder/'config.json').read_text(encoding='utf8'))
    records=unique(read_jsonl(folder/'predictions.jsonl'),'结果')
    verdicts={r['id']:r for r in metrics['details']} if metrics else {}
    calls={}
    for event in read_jsonl(folder/'calls.jsonl'):
        cid=event['call_id']
        if event['event']=='reserved':
            calls[cid]=dict(event)
        else:
            calls.setdefault(cid,{}).update(event)
    rows=[]
    for item in read_jsonl(folder/'dataset.jsonl'):
        rid=item['id']
        row={**item,**records.get(rid,{}),**verdicts.get(rid,{})}
        row['calls']=sorted((c for c in calls.values() if c.get('id')==rid),key=lambda c:c.get('local_index',0))
        rows.append(row)
    if metrics:
        summary=f"正确 {metrics['correct']}/{metrics['n']}，准确率 {metrics['accuracy']:.2%}；缺失 {metrics['missing']}，无效或缺失 {metrics['invalid']}；调用 {metrics['calls']}/{metrics['budget_limit']}。"
    else:
        summary=f"完成 {len(records)}/{config['dataset_n']} 题；尚未提供标签，未计算准确率。"
    payload={'name':folder.name,'config':config,'metrics':metrics,'summary':summary,'rows':rows}
    target=folder/'evaluations'/f'evaluation_{token}.html'
    target.parent.mkdir(exist_ok=True)
    encoded=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    target.write_text(DETAIL.replace('__STYLE__',STYLE).replace('__DATA__',encoded),encoding='utf8')
    return target

def leaderboard(root, open_browser=True):
    root=Path(root).resolve()
    groups={}
    for cpath in root.rglob('config.json'):
        config=json.loads(cpath.read_text(encoding='utf8'))
        if config.get('contract')!=CONTRACT:
            continue
        folder=cpath.parent
        snapshots=sorted((folder/'evaluations').glob('evaluation_*.json'))
        if not snapshots:
            continue
        saved=json.loads(snapshots[-1].read_text(encoding='utf8'))
        metrics=saved['metrics']
        key=(config['data_sha256'],config['examples_sha256'],config['deployment_sha256'],
             config['model'],bool(config['mock']),metrics.get('labels_sha256','') if metrics else '')
        groups.setdefault(key,[]).append({'name':folder.name,'config':config,'metrics':metrics,
              'detail':str(snapshots[-1].with_suffix('.html'))})
    token=stamp()
    out=root/'reports'
    out.mkdir(exist_ok=True)
    entries=[]
    sections=[]
    for key,items in sorted(groups.items(),key=lambda kv:(kv[0][4],kv[1][0]['config']['dataset_name'])):
        items.sort(key=lambda r:(-(r['metrics']['accuracy'] if r['metrics'] else -1),r['name']))
        name=items[0]['config']['dataset_name']
        heading=f"{name} · {key[3]} · "+('MOCK模拟（非正式成绩）' if key[4] else '模型运行')
        heading+=' · 部署配置 '+items[0]['config'].get('deployment_profile','旧记录')
        body=[]
        for item in items:
            m=item['metrics'];c=item['config']
            score=f"{m['accuracy']:.2%} ({m['correct']}/{m['n']})" if m else '未评测'
            status='完整' if m and m['submitted']==m['n'] and c['selected_n']==c['dataset_n'] else '小批/不完整'
            if m and m['invalid']:
                status+=f" · 无效/缺失 {m['invalid']}"
            link=quote(os.path.relpath(item['detail'],out).replace(os.sep,'/'),safe='/')
            body.append(f"<tr><td><a href='{link}'>{html.escape(item['name'])}</a></td><td>{score}</td><td>{m['calls'] if m else '—'}/{c['budget_limit']}</td><td>{html.escape(status)}</td></tr>")
        sections.append('<section><h2>'+html.escape(heading)+'</h2><p class=\"muted\">同一数据、示例池、部署配置与标签下按准确率降序。数据指纹 '+key[0][:12]+'</p><table><tr><th>运行记录（点击查看逐题过程）</th><th>准确率</th><th>调用数/预算</th><th>状态</th></tr>'+''.join(body)+'</table></section>')
        entries.append({'group':list(key),'runs':items})
    target=out/f'results_{token}.html'
    target.write_text('<!doctype html><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>全部实验结果</title><style>'+STYLE+'</style><header><h1>全部实验结果</h1><div>准确率从高到低 · 点击运行记录查看完整调用过程</div></header><main>'+(''.join(sections) or '<section>尚无新框架的评估记录。</section>')+'</main>',encoding='utf8')
    target.with_suffix('.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf8')
    print('结果汇总：'+str(target))
    if open_browser:
        webbrowser.open(target.as_uri())
    return target

def finish(folder, labels=None, root=None, open_browser=True):
    folder=Path(folder).resolve()
    metrics=evaluate(folder/'predictions.jsonl',labels) if labels else None
    token=stamp()
    detail=save_detail(folder,metrics,token)
    detail.with_suffix('.json').write_text(json.dumps({'evaluated_at':dt.datetime.now(dt.timezone.utc).isoformat(),
        'metrics':metrics},ensure_ascii=False,indent=2),encoding='utf8')
    leaderboard(root or folder.parent,open_browser)
    return metrics

if __name__=='__main__':
    argparse.ArgumentParser(description='重新汇总runs中的全部结果。').parse_args()
    leaderboard(ROOT/'runs', open_browser=False)
