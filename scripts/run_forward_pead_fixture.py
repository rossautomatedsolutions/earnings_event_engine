"""Strict bounded PEAD forward profile. Existing PEAD mathematics remain unchanged."""
import argparse
from pathlib import Path
import sys
import pandas as pd


def run(data, horizon):
    from forward_outcome_support import qualify, fields, require, positive, feature, package, window_status
    from earnings_event_engine.research.pead import calculate_announcement_reaction, calculate_forward_drift, assign_reaction_bucket
    require(horizon in (5, 10, 20, 40), 'FAMILY_SCOPE')
    rows = qualify(data); calendar = data['calendar']; features = []; events = {}
    for event in data['events']:
        fields(event, 'event_id subject event_date timing timing_source')
        # This synthetic profile records a named evidence source, not a timestamp
        # or a claim of historical public availability. No producer enum exists.
        # Require canonical printable text; never stringify arbitrary JSON.
        source = event['timing_source']
        require(type(source) is str and bool(source) and source == source.strip()
                and source.isprintable() and any(c.isalpha() for c in source),
                'EVENT_TIMING_UNQUALIFIED')
        require(event['timing'] in ('BMO', 'AMC'), 'EVENT_TIMING_UNQUALIFIED')
        require(event['event_date'] in calendar and event['subject'] in rows, 'EVENT_CALENDAR')
        i = calendar.index(event['event_date']) + (event['timing'] == 'AMC')
        require(0 < i < len(calendar), 'REACTION_SESSION_UNAVAILABLE')
        t, pre = calendar[i], calendar[i-1]; prices = rows[event['subject']]
        require(all(s in prices and positive(prices[s]['close']) for s in (pre, t)), 'REACTION_PRICE')
        key = event['subject'] + ':' + event['event_id']
        require(key not in events, 'DUPLICATE_OBSERVATION')
        frame = pd.DataFrame([{'event_id': key, 'ticker': event['subject'], 'event_date': event['event_date'],
                               'pre_close': prices[pre]['close'], 'post_close': prices[t]['close'],
                               'valid_event': True, 'price_alignment_valid': True}])
        reaction = assign_reaction_bucket(calculate_announcement_reaction(frame), threshold=.05)
        value = float(reaction.iloc[0]['reaction_return'])
        group = {'positive_reaction': 'positive', 'negative_reaction': 'negative', 'neutral_reaction': 'neutral'}[reaction.iloc[0]['reaction_bucket']]
        events[key] = frame
        features.append(feature(key, event['subject'], t, pre, value, group,
                                event['event_date'] + ':' + event['timing']))
    def label(f):
        prices = rows[f['subject']]
        end, reason = window_status(calendar, prices, f['session'], horizon)
        if reason:
            return end, None, reason
        frame = events[f['observation_id']].copy()
        frame[f'forward_close_{horizon}d'] = prices[end]['close']
        calculated = calculate_forward_drift(frame, windows=(horizon,))
        return end, float(calculated.iloc[0][f'forward_return_{horizon}d']), None
    return package(data, 'pead', 'reaction_state', horizon, 'raw_return', features, label)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--horizon', type=int, required=True); p.add_argument('--alpha-scripts', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root/'src'), str(root), a.alpha_scripts]
    from forward_outcome_support import read_json, write_result
    write_result(a.output, run(read_json(a.input), a.horizon))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
