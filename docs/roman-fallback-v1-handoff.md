# Roman FFT fallback v1 引き継ぎ資料

**作成日:** 2026-09-29  
**状態:** この検証はここで打ち切り。再実行はしていない。  
**対象:** 45イベントのフォールバック実行

## 結論

通常の FFT seed → short LM → long LM で苦戦したイベントに対し、保存済み
FFT結果から候補を広げて、複数seedを direct VBM + LM に通した。

フォールバック後の `chi2/dof` は、45件中38件が `<=1.1` まで改善した。
`>1.1` が7件残り、そのうち明確に悪い `>2` は次の5件だった。

```text
event    normal LM       fallback v1
9920076     53.2726          52.5098
9920050     28.5735          20.2533
9920061     64.4619           9.9393
9910043      9.7008           9.5977
9920088      3.3451           3.3422
```

この5件は「フォールバックでも原理的に不可能」とはまだ言えない。今回の
fallback v1ではPowellを無効にしており、`powell_ran=false` が45件すべてで
ある。したがって、未解決の主因が探索不足なのか、座標・モデルの局所解なのか
はまだ切り分けていない。

`1.1 < chi2/dof <= 2` の残り2件は次の通り。

```text
9910011  1.5011
9910082  1.1329
```

## 入力と結果ファイル

フォールバックの集計は
[`results/roman_local/roman_fft_fallback_v1_batch45/summary.json`](../results/roman_local/roman_fft_fallback_v1_batch45/summary.json)
にある。45件それぞれの結果は同じディレクトリの `events/`、short/long LMの
チェックポイントは `checkpoints/` に保存されている。

比較に使った通常経路の結果は
[`results/roman_local/roman_alpha_direct_short_long_2048_vbmfix_dchi2gt100_batch176`](../results/roman_local/roman_alpha_direct_short_long_2048_vbmfix_dchi2gt100_batch176)
である。

失敗イベントの診断画像は
[`assets/roman-failure-events-chi2gt1p1/README.md`](../assets/roman-failure-events-chi2gt1p1/README.md)
にまとめてある。これは通常LMの画像であり、fallbackモデルを重ねた図ではない。

## フォールバックの手順

### 1. 保存済みFFT結果から候補を再展開

全マップFFTをやり直すのではなく、各イベントの既存FFT report/minimaを入力にした。

- 上位マップ数: `1024`
- 既存の16点geometry stencilをgeometryごとに再展開
- alpha解像度: `n_alpha=2048`
- alpha local-window mode: 有効
- 再展開候補数: `1024 x 16 = 16,384`
- truthによる候補選択: なし

候補の重複を減らすため、各geometryについて以下を残した。

- elite candidates: 12
- geometry/grid上でdiverseな候補: 4
- 最小grid距離: 2.0

これにより、最終的に各イベント256 seedを選んだ。

### 2. 全seedをshort LM

各seedを direct VBMicrolensing で評価した後、7パラメータをLMで最適化した。

```text
(t0, u0, tE, s, q, rho, alpha)
```

設定は次の通り。

- short LMの最大評価回数: `80`
- `short_stop_chi2_dof`: `1.2`
- stop判定に必要な最低完了数: `16`
- source flux / blend flux: 各残差評価で解析的にprofile
- magnification: direct VBMicrolensing

### 3. 有望seedをlong LMへ継続

short LM後、次の候補をlong LMへ回した。

- `continue_best`: 8
- 改善量の大きい候補: 最大8
- 実際のcontinuation数: イベントごとに8--28
- long LMの最大評価回数: `320`

最終的なイベントごとのcontinuation総数は766だった。

## Powellについて

設定上は

```text
powell_trigger_chi2_dof = 2.0
powell_enabled = false
```

となっていた。つまり、`chi2/dof > 2` の候補をPowellへ送る条件は記録されて
いたが、Powell本体は無効化されていた。結果JSONでも45件すべて
`powell_ran=false`、`powell=[]` である。

したがって、残った5件について「Powellでも無理だった」という意味ではない。
Powellを使わず、候補bankとLMだけで打ち切った結果である。

## 結果の内訳

```text
対象イベント                         45
chi2/dof <= 1.0                       11
chi2/dof <= 1.1                       38
chi2/dof <= 2.0                       40
chi2/dof <= 5.0                       41
chi2/dof <= 10                        43
chi2/dof > 2                           5
chi2/dof > 1.1                         7
```

fallback前後で特に改善した例は次の通り。

```text
9920061   64.4619 ->  9.9393
9920062  446.2595 ->  1.0134
9920069 1569.5178 ->  1.0036
9920070 1367.0862 ->  0.9971
9920080  774.3634 ->  1.0202
9920081 1502.4531 ->  1.0674
9920092 1133.2977 ->  0.9965
```

一方、`9920076` と `9920050` はbankを広げてもほとんど改善しなかった。
`9920061` は大幅に改善したものの、まだ `chi2/dof≈9.94` が残っている。

### 全45件の最終値

以下はfallback v1の `best_chi2_dof` を悪い順に並べたもの。

```text
9920076  52.5098
9920050  20.2533
9920061   9.9393
9910043   9.5977
9920088   3.3422
9910011   1.5011
9910082   1.1329
9920081   1.0674
9920089   1.0401
9920072   1.0208
9920080   1.0202
9920097   1.0195
9920027   1.0164
9920062   1.0134
9920044   1.0131
9910073   1.0128
9920007   1.0115
9920017   1.0108
9920003   1.0105
9920083   1.0092
9920006   1.0083
9920098   1.0082
9920029   1.0079
9920096   1.0069
9920049   1.0064
9920075   1.0044
9920069   1.0036
9910036   1.0032
9910099   1.0029
9920099   1.0014
9910063   1.0013
9920095   1.0006
9920021   1.0002
9910094   0.9983
9920065   0.9974
9920004   0.9973
9920039   0.9973
9920070   0.9971
9920092   0.9965
9920079   0.9948
9920068   0.9928
9920074   0.9923
9920087   0.9881
9920078   0.9812
```

## 計算時間

45イベントのevent-level `wall_seconds` の合計は約17.6時間だった。これは
並列実行中の各イベント時間の合計であり、実際の経過時間ではない。

- 最短イベント: 約3.6分
- 中央値: 約13.5分
- 最長イベント: 約82.4分
- fallback設定: `workers=3`, `event_workers=6`

## 解釈と限界

1. フォールバック候補の選択自体はtruth blindだった。truthとの距離で候補を
   選んだり、truthに近い `s,q,alpha,rho` をseedへ入れたりしていない。
2. ただし、上流のFFT geometry stencilはtruth-centeredな`t0,u0,tE`を使う
   実験設計である。したがって、実験全体を完全blind recoveryとは呼ばない。
3. `alpha`は`u0`の符号と結び付いた約πの表現差があり得るため、alphaの数値差
   だけで失敗とは判定しない。
4. 今回は上位1024 mapまでしか再展開していない。正しい谷が保存FFT上位に
   入っていなければ、このfallbackでは回収できない。
5. geometryごとに16点の候補bankを使ったが、連続geometryのglobal searchでは
   ない。`tE`や`rho`を含む局所谷から抜けられない可能性が残る。
6. `9920076`, `9920050`, `9920061`, `9910043`, `9920088` はPowell未実行
   のまま残った5件であり、現時点では「探索不能」ではなく「v1未解決」と記録
   するのが正確である。

## 再開する場合の候補

この検証は打ち切りとするが、将来再開する場合は次の順が自然である。

1. 残った7件だけを対象にする。
2. `chi2/dof > 2` の5件を優先する。
3. 既存のfallback bestを初期点に、Powellを全7パラメータへ適用する。
4. Powell後にdirect VBM long LMを行う。
5. `s,q`だけ固定・最適化するのではなく、`t0,u0,tE,s,q,rho,alpha`全体を
   同じ座標契約で動かす。

この次段階は今回の成果には含めていない。
