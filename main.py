import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# 🔧 إعداد اللوج الاحترافي
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class TradeSignal:
    direction: TradeDirection
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: int

class SmartTradingCouncil:
    """🧠 مجلس الإدارة الذكي لاتخاذ قرارات التداول"""
    
    def __init__(self):
        self.members = {
            'macd_analyst': {'weight': 0.25, 'last_vote': None},
            'vwap_analyst': {'weight': 0.25, 'last_vote': None},
            'volume_analyst': {'weight': 0.25, 'last_vote': None},
            'price_action_analyst': {'weight': 0.25, 'last_vote': None}
        }
        self.min_confidence = 0.75
        self.last_decision = None
    
    def analyze_macd(self, data: pd.DataFrame) -> Dict:
        """📊 تحليل MACD المتقدم"""
        try:
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]
            prev_histogram = histogram.iloc[-2]
            
            # قرار MACD
            if current_macd > current_signal and current_histogram > 0 and current_histogram > prev_histogram:
                return {'vote': 'BUY', 'confidence': min(0.9, abs(current_histogram) * 10), 'reason': 'MACD صعودي قوي'}
            elif current_macd < current_signal and current_histogram < 0 and current_histogram < prev_histogram:
                return {'vote': 'SELL', 'confidence': min(0.9, abs(current_histogram) * 10), 'reason': 'MACD هبوطي قوي'}
            else:
                return {'vote': 'HOLD', 'confidence': 0.3, 'reason': 'MACD محايد'}
                
        except Exception as e:
            logger.error(f"خطأ في تحليل MACD: {e}")
            return {'vote': 'HOLD', 'confidence': 0.1, 'reason': 'خطأ تقني'}
    
    def analyze_vwap(self, data: pd.DataFrame) -> Dict:
        """📊 تحليل VWAP مع الزخم"""
        try:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
            
            current_price = data['close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            price_vwap_ratio = current_price / current_vwap
            
            # قرار VWAP
            if current_price > current_vwap and price_vwap_ratio > 1.002:
                return {'vote': 'BUY', 'confidence': min(0.85, (price_vwap_ratio - 1) * 100), 'reason': 'السعر فوق VWAP مع زخم'}
            elif current_price < current_vwap and price_vwap_ratio < 0.998:
                return {'vote': 'SELL', 'confidence': min(0.85, (1 - price_vwap_ratio) * 100), 'reason': 'السعر تحت VWAP مع زخم'}
            else:
                return {'vote': 'HOLD', 'confidence': 0.4, 'reason': 'السعر قريب من VWAP'}
                
        except Exception as e:
            logger.error(f"خطأ في تحليل VWAP: {e}")
            return {'vote': 'HOLD', 'confidence': 0.1, 'reason': 'خطأ تقني'}
    
    def analyze_volume_delta(self, data: pd.DataFrame) -> Dict:
        """📊 تحليل Delta Volume"""
        try:
            if 'buy_volume' not in data.columns or 'sell_volume' not in data.columns:
                return {'vote': 'HOLD', 'confidence': 0.2, 'reason': 'بيانات Volume غير متوفرة'}
            
            current_buy_volume = data['buy_volume'].iloc[-1]
            current_sell_volume = data['sell_volume'].iloc[-1]
            delta = current_buy_volume - current_sell_volume
            total_volume = current_buy_volume + current_sell_volume
            
            if total_volume > 0:
                delta_ratio = delta / total_volume
            else:
                delta_ratio = 0
            
            # قرار Volume Delta
            if delta_ratio > 0.1:
                return {'vote': 'BUY', 'confidence': min(0.8, abs(delta_ratio) * 5), 'reason': 'حجم شرائي قوي'}
            elif delta_ratio < -0.1:
                return {'vote': 'SELL', 'confidence': min(0.8, abs(delta_ratio) * 5), 'reason': 'حجم بيعي قوي'}
            else:
                return {'vote': 'HOLD', 'confidence': 0.3, 'reason': 'حجم متوازن'}
                
        except Exception as e:
            logger.error(f"خطأ في تحليل Volume: {e}")
            return {'vote': 'HOLD', 'confidence': 0.1, 'reason': 'خطأ تقني'}
    
    def analyze_price_action(self, data: pd.DataFrame) -> Dict:
        """📊 تحليل Price Action المتقدم"""
        try:
            current_close = data['close'].iloc[-1]
            prev_close = data['close'].iloc[-2]
            current_high = data['high'].iloc[-1]
            current_low = data['low'].iloc[-1]
            prev_high = data['high'].iloc[-2]
            prev_low = data['low'].iloc[-2]
            
            # اكتشاف الانعكاسات والاستمرارية
            price_change = (current_close - prev_close) / prev_close * 100
            
            # قرار Price Action
            if current_close > prev_high and price_change > 0.1:
                return {'vote': 'BUY', 'confidence': min(0.9, abs(price_change) * 2), 'reason': 'كسر مقاومة مع زخم'}
            elif current_close < prev_low and price_change < -0.1:
                return {'vote': 'SELL', 'confidence': min(0.9, abs(price_change) * 2), 'reason': 'كسر دعم مع زخم'}
            elif price_change > 0.05:
                return {'vote': 'BUY', 'confidence': 0.6, 'reason': 'زخم صعودي'}
            elif price_change < -0.05:
                return {'vote': 'SELL', 'confidence': 0.6, 'reason': 'زخم هبوطي'}
            else:
                return {'vote': 'HOLD', 'confidence': 0.4, 'reason': 'سوق جانبي'}
                
        except Exception as e:
            logger.error(f"خطأ في تحليل Price Action: {e}")
            return {'vote': 'HOLD', 'confidence': 0.1, 'reason': 'خطأ تقني'}
    
    def convene_meeting(self, data: pd.DataFrame) -> Optional[TradeSignal]:
        """🧠 عقد اجتماع مجلس الإدارة واتخاذ القرار"""
        logger.info("🧠 مجلس الإدارة يعقد اجتماعًا لاتخاذ قرار التداول...")
        
        # جمع آراء الأعضاء
        votes = {}
        total_confidence = 0
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0
        
        # تحليل MACD
        macd_analysis = self.analyze_macd(data)
        votes['macd'] = macd_analysis
        self.members['macd_analyst']['last_vote'] = macd_analysis
        
        # تحليل VWAP
        vwap_analysis = self.analyze_vwap(data)
        votes['vwap'] = vwap_analysis
        self.members['vwap_analyst']['last_vote'] = vwap_analysis
        
        # تحليل Volume Delta
        volume_analysis = self.analyze_volume_delta(data)
        votes['volume'] = volume_analysis
        self.members['volume_analyst']['last_vote'] = volume_analysis
        
        # تحليل Price Action
        price_action_analysis = self.analyze_price_action(data)
        votes['price_action'] = price_action_analysis
        self.members['price_action_analyst']['last_vote'] = price_action_analysis
        
        # حساب الأصوات المرجحة
        weighted_votes = {
            'BUY': 0,
            'SELL': 0, 
            'HOLD': 0
        }
        
        for member, analysis in votes.items():
            weight = self.members[member + '_analyst']['weight']
            vote = analysis['vote']
            confidence = analysis['confidence']
            
            weighted_votes[vote] += weight * confidence
            
            if vote == 'BUY':
                buy_votes += 1
            elif vote == 'SELL':
                sell_votes += 1
            else:
                hold_votes += 1
        
        # اتخاذ القرار النهائي
        current_price = data['close'].iloc[-1]
        atr = self.calculate_atr(data)
        
        if weighted_votes['BUY'] >= self.min_confidence and buy_votes >= 2:
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 3)
            reason = " | ".join([v['reason'] for k, v in votes.items() if v['vote'] == 'BUY'])
            
            signal = TradeSignal(
                direction=TradeDirection.LONG,
                confidence=weighted_votes['BUY'],
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
                timestamp=int(time.time())
            )
            
            logger.info(f"✅ قرار مجلس الإدارة: فتح LONG | ثقة: {signal.confidence:.2f}")
            logger.info(f"   📍 الدخول: {signal.entry_price:.4f} | 🛑 وقف: {signal.stop_loss:.4f} | 🎯 هدف: {signal.take_profit:.4f}")
            logger.info(f"   📝 الأسباب: {signal.reason}")
            
            self.last_decision = signal
            return signal
            
        elif weighted_votes['SELL'] >= self.min_confidence and sell_votes >= 2:
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 3)
            reason = " | ".join([v['reason'] for k, v in votes.items() if v['vote'] == 'SELL'])
            
            signal = TradeSignal(
                direction=TradeDirection.SHORT,
                confidence=weighted_votes['SELL'],
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
                timestamp=int(time.time())
            )
            
            logger.info(f"✅ قرار مجلس الإدارة: فتح SHORT | ثقة: {signal.confidence:.2f}")
            logger.info(f"   📍 الدخول: {signal.entry_price:.4f} | 🛑 وقف: {signal.stop_loss:.4f} | 🎯 هدف: {signal.take_profit:.4f}")
            logger.info(f"   📝 الأسباب: {signal.reason}")
            
            self.last_decision = signal
            return signal
        else:
            logger.info(f"❌ مجلس الإدارة قرر الانتظار | أصوات شراء: {buy_votes} | أصوات بيع: {sell_votes}")
            logger.info(f"   📊 الثقة: شراء {weighted_votes['BUY']:.2f} | بيع {weighted_votes['SELL']:.2f}")
            return None
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """حساب Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr

class ProfessionalTradingBot:
    """🤖 البوت المتداول المحترف مع مجلس الإدارة الذكي"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.council = SmartTradingCouncil()
        self.active_trades = []
        self.trade_history = []
        self.last_candle_timestamp = None
        self.entry_filter = set()  # لمنع التكرار في نفس الشمعة
        
        logger.info("🤖 البوت المتداول المحترف يعمل بنظام مجلس الإدارة الذكي")
    
    def analyze_market(self, data: pd.DataFrame) -> Optional[TradeSignal]:
        """📈 تحليل السوق والبحث عن مناطق قوية"""
        
        # 🔁 فلترة لمنع التكرار في نفس الشمعة
        current_candle = data.index[-1]
        if current_candle in self.entry_filter:
            return None
        
        # 🧠 عقد اجتماع مجلس الإدارة لاتخاذ القرار
        signal = self.council.convene_meeting(data)
        
        if signal:
            self.entry_filter.add(current_candle)
            
        return signal
    
    def manage_open_trades(self, data: pd.DataFrame):
        """📊 إدارة الصفقات المفتوحة بشكل احترافي"""
        current_price = data['close'].iloc[-1]
        
        for trade in self.active_trades[:]:
            # تحديث بيانات الصفقة
            trade['current_price'] = current_price
            trade['pnl'] = self.calculate_pnl(trade, current_price)
            
            # 📉 غلق صارم عند الانعكاس
            if self.should_close_trade(trade, data):
                self.close_trade(trade, "إغلاق صارم - انعكاس الإشارة")
            
            # 💰 جني الأرباح الذكي
            elif self.should_take_profit(trade, data):
                self.close_trade(trade, "جني أرباح ذكي")
            
            # 🛑 وقف الخسارة
            elif self.is_stop_loss_hit(trade, current_price):
                self.close_trade(trade, "وقف خسارة")
    
    def should_close_trade(self, trade: Dict, data: pd.DataFrame) -> bool:
        """📉 قرار الإغلاق الصارم"""
        # تحليل انعكاس الإشارة
        current_signal = self.council.convene_meeting(data)
        
        if current_signal:
            if trade['direction'] == TradeDirection.LONG and current_signal.direction == TradeDirection.SHORT:
                return True
            elif trade['direction'] == TradeDirection.SHORT and current_signal.direction == TradeDirection.LONG:
                return True
        
        return False
    
    def should_take_profit(self, trade: Dict, data: pd.DataFrame) -> bool:
        """💰 قرار جني الأرباح الذكي"""
        current_price = data['close'].iloc[-1]
        
        if trade['direction'] == TradeDirection.LONG:
            profit_ratio = (current_price - trade['entry_price']) / trade['entry_price']
            # جني جزئي عند 1.5% وكلّي عند 3%
            if profit_ratio >= 0.03:
                return True
        else:  # SHORT
            profit_ratio = (trade['entry_price'] - current_price) / trade['entry_price']
            if profit_ratio >= 0.03:
                return True
        
        return False
    
    def is_stop_loss_hit(self, trade: Dict, current_price: float) -> bool:
        """🛑 التحقق من وقف الخسارة"""
        if trade['direction'] == TradeDirection.LONG:
            return current_price <= trade['stop_loss']
        else:  # SHORT
            return current_price >= trade['stop_loss']
    
    def calculate_pnl(self, trade: Dict, current_price: float) -> float:
        """حساب الربح/الخسارة"""
        if trade['direction'] == TradeDirection.LONG:
            return (current_price - trade['entry_price']) / trade['entry_price'] * 100
        else:  # SHORT
            return (trade['entry_price'] - current_price) / trade['entry_price'] * 100
    
    def execute_trade(self, signal: TradeSignal):
        """💼 تنفيذ الصفقة"""
        trade = {
            'id': len(self.trade_history) + 1,
            'direction': signal.direction,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'entry_time': signal.timestamp,
            'reason': signal.reason,
            'confidence': signal.confidence,
            'status': 'OPEN'
        }
        
        self.active_trades.append(trade)
        self.trade_history.append(trade.copy())
        
        logger.info(f"🎯 تم فتح صفقة {trade['direction'].value} #{trade['id']}")
        logger.info(f"   💰 السعر: {trade['entry_price']:.4f}")
        logger.info(f"   🛑 وقف: {trade['stop_loss']:.4f}")
        logger.info(f"   🎯 هدف: {trade['take_profit']:.4f}")
        logger.info(f"   📊 الثقة: {trade['confidence']:.2f}")
    
    def close_trade(self, trade: Dict, reason: str):
        """🔚 إغلاق الصفقة"""
        trade['exit_time'] = int(time.time())
        trade['exit_price'] = trade['current_price']
        trade['final_pnl'] = trade['pnl']
        trade['status'] = 'CLOSED'
        trade['close_reason'] = reason
        
        self.active_trades.remove(trade)
        
        # تحديث السجل
        for hist_trade in self.trade_history:
            if hist_trade['id'] == trade['id']:
                hist_trade.update(trade)
                break
        
        logger.info(f"🔚 تم إغلاق صفقة #{trade['id']}")
        logger.info(f"   📊 النتيجة: {trade['final_pnl']:+.2f}%")
        logger.info(f"   📝 السبب: {reason}")
    
    def get_performance_report(self) -> Dict:
        """📈 تقرير أداء البوت"""
        if not self.trade_history:
            return {}
        
        closed_trades = [t for t in self.trade_history if t['status'] == 'CLOSED']
        
        if not closed_trades:
            return {}
        
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t['final_pnl'] > 0])
        losing_trades = len([t for t in closed_trades if t['final_pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = sum(t['final_pnl'] for t in closed_trades)
        avg_pnl = total_pnl / total_trades
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'average_pnl': avg_pnl
        }

# نموذج استخدام البوت
def main():
    # تهيئة البوت
    bot = ProfessionalTradingBot(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET")
    
    # محاكاة بيانات السوق
    while True:
        try:
            # جلب بيانات السوق (هنا مثال بمحاكاة)
            market_data = simulate_market_data()
            
            # 📈 البحث عن مناطق قوية واتخاذ القرار
            signal = bot.analyze_market(market_data)
            
            if signal and not bot.active_trades:
                bot.execute_trade(signal)
            
            # 📊 إدارة الصفقات المفتوحة
            if bot.active_trades:
                bot.manage_open_trades(market_data)
            
            # 📈 عرض تقرير الأداء
            if len(bot.trade_history) % 10 == 0:
                report = bot.get_performance_report()
                if report:
                    logger.info("📈 تقرير أداء البوت:")
                    logger.info(f"   📊 إجمالي الصفقات: {report['total_trades']}")
                    logger.info(f"   ✅ صفقات رابحة: {report['winning_trades']}")
                    logger.info(f"   ❌ صفقات خاسرة: {report['losing_trades']}")
                    logger.info(f"   🎯 نسبة النجاح: {report['win_rate']:.1f}%")
                    logger.info(f"   💰 إجمالي الأرباح: {report['total_pnl']:.2f}%")
            
            time.sleep(60)  # انتظار دقيقة بين التحليلات
            
        except Exception as e:
            logger.error(f"خطأ في التشغيل: {e}")
            time.sleep(10)

def simulate_market_data() -> pd.DataFrame:
    """محاكاة بيانات السوق للاختبار"""
    # هذه دالة للمثال فقط - يجب استبدالها ببيانات حقيقية
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.normal(1000, 100, 100),
        'buy_volume': np.random.normal(500, 50, 100),
        'sell_volume': np.random.normal(500, 50, 100)
    }, index=dates)
    
    return data

if __name__ == "__main__":
    main()
