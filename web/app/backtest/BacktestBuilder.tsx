"use client";

import { useState } from "react";

type BacktestTicker = {
  id: string;
  ticker: string;
};

type BacktestGroup = {
  id: string;
  name: string;
  weight: number;
  tickers: BacktestTicker[];
};

const PERIOD_OPTIONS = Array.from({ length: 24 }, (_, index) => index + 1);

let groupCounter = 1;
let tickerCounter = 1;

function createTicker(ticker = ""): BacktestTicker {
  return {
    id: `ticker-${tickerCounter++}`,
    ticker,
  };
}

function createGroup(): BacktestGroup {
  const nextNumber = groupCounter++;
  return {
    id: `group-${nextNumber}`,
    name: `그룹${nextNumber}`,
    weight: 10,
    tickers: [createTicker()],
  };
}

function getTotalWeight(groups: BacktestGroup[]): number {
  return groups.reduce((sum, group) => sum + group.weight, 0);
}

export function BacktestBuilder() {
  const [periodMonths, setPeriodMonths] = useState(12);
  const [groups, setGroups] = useState<BacktestGroup[]>([createGroup()]);
  const [hasExecuted, setHasExecuted] = useState(false);

  function updateGroup(groupId: string, updater: (group: BacktestGroup) => BacktestGroup) {
    setGroups((current) => current.map((group) => (group.id === groupId ? updater(group) : group)));
  }

  function handleAddGroup() {
    setGroups((current) => [...current, createGroup()]);
  }

  function handleDeleteGroup(groupId: string) {
    setGroups((current) => current.filter((group) => group.id !== groupId));
  }

  function handleAddTicker(groupId: string) {
    updateGroup(groupId, (group) => ({
      ...group,
      tickers: [...group.tickers, createTicker()],
    }));
  }

  function handleDeleteTicker(groupId: string, tickerId: string) {
    updateGroup(groupId, (group) => ({
      ...group,
      tickers: group.tickers.filter((ticker) => ticker.id !== tickerId),
    }));
  }

  const totalWeight = getTotalWeight(groups);

  return (
    <div className="appPageStack">
      <div className="card appCard">
        <div className="card-body appCardBodyTight">
          <div className="backtestToolbar">
            <div className="backtestToolbarLeft">
              <div>
                <div className="subheader">기간</div>
                <select
                  className="form-select"
                  value={periodMonths}
                  onChange={(event) => setPeriodMonths(Number(event.target.value))}
                >
                  {PERIOD_OPTIONS.map((months) => (
                    <option key={months} value={months}>
                      최근 {months}달
                    </option>
                  ))}
                </select>
              </div>
              <div className="backtestSummaryText">
                그룹 {groups.length}개 · 총 비중 {totalWeight}%
              </div>
            </div>
            <div className="backtestToolbarActions">
              <button type="button" className="btn btn-outline-secondary" onClick={handleAddGroup}>
                그룹 추가
              </button>
              <button type="button" className="btn btn-primary" onClick={() => setHasExecuted(true)}>
                백테스트 실행
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="backtestGroupGrid">
        {groups.map((group, index) => (
          <div key={group.id} className="card appCard backtestGroupCard">
            <div className="card-header appCardHeader">
              <div>
                <h3 className="card-title">{group.name}</h3>
                <div className="tableMuted">그룹 {index + 1}</div>
              </div>
              <button
                type="button"
                className="btn btn-outline-danger btn-sm"
                onClick={() => handleDeleteGroup(group.id)}
                disabled={groups.length === 1}
              >
                그룹 삭제
              </button>
            </div>
            <div className="card-body appCardBody backtestGroupBody">
              <div className="backtestFieldGrid">
                <label className="form-label mb-0">
                  <span className="subheader">그룹명</span>
                  <input
                    type="text"
                    className="form-control"
                    value={group.name}
                    onChange={(event) =>
                      updateGroup(group.id, (currentGroup) => ({
                        ...currentGroup,
                        name: event.target.value,
                      }))
                    }
                  />
                </label>
                <label className="form-label mb-0">
                  <span className="subheader">비중(%)</span>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    className="form-control"
                    value={group.weight}
                    onChange={(event) =>
                      updateGroup(group.id, (currentGroup) => ({
                        ...currentGroup,
                        weight: Number(event.target.value || 0),
                      }))
                    }
                  />
                </label>
              </div>

              <div className="backtestTickerSection">
                <div className="d-flex align-items-center justify-content-between">
                  <div className="subheader">ETF 티커</div>
                  <button type="button" className="btn btn-outline-secondary btn-sm" onClick={() => handleAddTicker(group.id)}>
                    종목 추가
                  </button>
                </div>
                <div className="backtestTickerList">
                  {group.tickers.map((ticker, tickerIndex) => (
                    <div key={ticker.id} className="backtestTickerRow">
                      <span className="tableMuted backtestTickerIndex">{tickerIndex + 1}</span>
                      <input
                        type="text"
                        className="form-control"
                        placeholder="예: 069500"
                        value={ticker.ticker}
                        onChange={(event) =>
                          updateGroup(group.id, (currentGroup) => ({
                            ...currentGroup,
                            tickers: currentGroup.tickers.map((currentTicker) =>
                              currentTicker.id === ticker.id
                                ? { ...currentTicker, ticker: event.target.value.toUpperCase() }
                                : currentTicker,
                            ),
                          }))
                        }
                      />
                      <button
                        type="button"
                        className="btn btn-outline-danger btn-sm"
                        onClick={() => handleDeleteTicker(group.id, ticker.id)}
                        disabled={group.tickers.length === 1}
                      >
                        삭제
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card appCard">
        <div className="card-header appCardHeader">
          <h3 className="card-title">결과</h3>
        </div>
        <div className="card-body appCardBody">
          {hasExecuted ? (
            <div className="text-secondary" style={{ fontSize: "0.95rem" }}>
              백테스트 계산 결과는 다음 단계에서 연결한다.
            </div>
          ) : (
            <div className="text-secondary" style={{ fontSize: "0.95rem" }}>
              설정을 마친 뒤 <strong>백테스트 실행</strong> 버튼을 눌러 결과를 확인한다.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
