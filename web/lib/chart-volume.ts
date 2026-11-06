/** 차트 아래 거래량 막대 — `/ticker` 상세와 전략 차트(`HoldingChart`)가 함께 쓴다.
 *
 *  거래량은 별도 가격축(`volume`)에 그려 차트 아래 20% 를 쓴다. 캔들 축은
 *  `PRICE_SCALE_BOTTOM_WITH_VOLUME` 만큼 아래를 비워 두 영역이 겹치지 않게 한다.
 *  막대 색은 전일 종가 대비 상승(빨강)·하락(파랑)이다.
 *
 *  기본 히스토그램(`HistogramSeries`)을 쓰지 않고 막대를 직접 그린다. 기본 히스토그램은
 *  봉 사이에 **항상 1px 틈**을 둬서, 봉 간격이 좁은 전략 카드(약 270봉·510px, 간격 1.9px)
 *  에서는 막대 1px + 틈 1px 로 면적의 절반이 흰 배경이 돼 흐리게 보였다. 여기서는 간격이
 *  좁으면 틈 없이 붙여 그리고, 넓으면(`/ticker` 상세, 간격 약 5px) 전처럼 1px 틈을 둔다.
 */

import { customSeriesDefaultOptions } from "lightweight-charts";
import type {
  CustomData,
  CustomSeriesOptions,
  CustomSeriesWhitespaceData,
  IChartApi,
  ICustomSeriesPaneRenderer,
  ICustomSeriesPaneView,
  PaneRendererCustomData,
  PriceToCoordinateConverter,
  Time,
} from "lightweight-charts";

/** 캔들 가격축이 비워 둘 아래 여백 — 거래량 영역(아래 20%)과 겹치지 않게 한다. */
export const PRICE_SCALE_BOTTOM_WITH_VOLUME = 0.25;

const VOLUME_PANE_TOP = 0.8;
// 캔들과 같은 색을 불투명하게 쓴다 — 거래량은 캔들과 겹치지 않는 아래 영역에 따로 그린다.
const VOLUME_UP_COLOR = "#e03131";
const VOLUME_DOWN_COLOR = "#206bc4";
/** 봉 간격(CSS px)이 이 값 이상일 때만 막대 사이에 틈을 둔다. */
const GAP_MIN_BAR_SPACING = 3;

export type VolumeRow = { time: string; close: number | null; volume: number | null };

type VolumeBar = CustomData<Time> & { value: number; color: string };
type DrawTarget = Parameters<ICustomSeriesPaneRenderer["draw"]>[0];

class VolumeColumnsView implements ICustomSeriesPaneView<Time, VolumeBar, CustomSeriesOptions> {
  private data: PaneRendererCustomData<Time, VolumeBar> | null = null;

  renderer(): ICustomSeriesPaneRenderer {
    return { draw: (target, priceToCoordinate) => this.draw(target, priceToCoordinate) };
  }

  update(data: PaneRendererCustomData<Time, VolumeBar>): void {
    this.data = data;
  }

  /** 자동 축 범위에 0 을 넣고, 마지막 값(현재값·점선 위치)은 거래량이다. */
  priceValueBuilder(row: VolumeBar): number[] {
    return [0, row.value];
  }

  isWhitespace(row: VolumeBar | CustomSeriesWhitespaceData<Time>): row is CustomSeriesWhitespaceData<Time> {
    return (row as Partial<VolumeBar>).value === undefined;
  }

  defaultOptions(): CustomSeriesOptions {
    return { ...customSeriesDefaultOptions, color: VOLUME_UP_COLOR };
  }

  private draw(target: DrawTarget, priceToCoordinate: PriceToCoordinateConverter): void {
    const data = this.data;
    const range = data?.visibleRange;
    if (!data || !range) return;
    const baseY = priceToCoordinate(0);
    if (baseY === null) return;

    target.useBitmapCoordinateSpace(({ context, horizontalPixelRatio, verticalPixelRatio }) => {
      const barSpacing = data.barSpacing * data.conflationFactor;
      const spacing = barSpacing * horizontalPixelRatio;
      const gap = barSpacing >= GAP_MIN_BAR_SPACING ? Math.max(1, Math.floor(horizontalPixelRatio)) : 0;
      const width = Math.max(1, Math.round(spacing) - gap);
      const bottom = Math.round(baseY * verticalPixelRatio);
      for (let i = range.from; i < range.to; i++) {
        const bar = data.bars[i];
        if (!bar) continue;
        const y = priceToCoordinate(bar.originalData.value);
        if (y === null) continue;
        const top = Math.round(y * verticalPixelRatio);
        const left = Math.round(bar.x * horizontalPixelRatio) - Math.floor(width / 2);
        // 색은 `barColor` 에서 읽는다 — 라이브러리가 데이터의 `color` 를 여기로 옮기고
        // `originalData` 에서는 빼 버린다(읽으면 undefined → 캔버스 기본값 검정).
        context.fillStyle = bar.barColor;
        context.fillRect(left, top, width, Math.max(1, bottom - top));
      }
    });
  }
}

export function addVolumeHistogram(chart: IChartApi, rows: VolumeRow[]): void {
  // 마지막 거래량 **라벨**만 끈다 — 가격축 라벨과 겹쳐 가격처럼 읽힌다. 점선은 남긴다.
  const series = chart.addCustomSeries(new VolumeColumnsView(), {
    priceFormat: { type: "volume" },
    priceScaleId: "volume",
    lastValueVisible: false,
  });
  chart.priceScale("volume").applyOptions({ scaleMargins: { top: VOLUME_PANE_TOP, bottom: 0 } });

  // 색은 **바로 앞 봉의 종가**와 비교한다. 거래량이 없는 봉을 거른 뒤 순번으로 앞 봉을 찾으면
  // 거른 봉만큼 어긋나므로, 거르기 전 순서에서 직전 종가를 들고 간다.
  const data: VolumeBar[] = [];
  let prevClose: number | null = null;
  for (const row of rows) {
    if (row.volume !== null && row.close !== null) {
      const isUp = prevClose === null || row.close >= prevClose;
      data.push({ time: row.time as Time, value: row.volume, color: isUp ? VOLUME_UP_COLOR : VOLUME_DOWN_COLOR });
    }
    if (row.close !== null) prevClose = row.close;
  }
  series.setData(data);
}
