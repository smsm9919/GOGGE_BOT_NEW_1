
def log_ai_trading_status(df: pd.DataFrame, indicators: Dict, decision: Dict, market_analysis: Dict):
    """تسجيل حالة التداول بالذكاء الاصطناعي"""
    current_price = df['close'].iloc[-1]
    volume_ratio = indicators.get('volume_ratio', 1)
    adx = indicators.get('adx', 0)
    rsi = indicators.get('rsi', 50)

    # معلومات المركز
    position_info = "⚪ لا توجد مراكز مفتوحة"
    if position_manager.positions:
        position = list(position_manager.positions.values())[0]
        profit_color = "green" if position['current_profit'] > 0 else "red"
        profit = f"{position['current_profit']:.2f}%"
        profit_colored = colored(profit, profit_color)
        position_info = f"{'🟢 شراء' if position['direction'] == 'LONG' else '🔴 بيع'} | الربح: {profit_colored}"

    # معلومات قرار الذكاء الاصطناعي
    action_color = "green" if decision['action'] == 'ENTER' else "red" if decision['action'] == 'EXIT' else "yellow"
    confidence_level = "🟢 عالي" if decision['confidence'] > 0.8 else "🟡 متوسط" if decision['confidence'] > 0.6 else "🔴 منخفض"

    # عرض الحالة المحترف
    print("\n" + "="*120)
    print(colored(f"🤖 بوت التداول بالذكاء الاصطناعي | {SYMBOL} | {INTERVAL} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "cyan", attrs=['bold']))
    print("="*120)

    # بيانات السوق
    print(colored("📈 تحليل السوق:", "white", attrs=['bold']))
    print(f"   💰 السعر: {current_price:.6f} | الحجم: {volume_ratio:.1f}x | "
          f"النطاق: {(df['high'].iloc[-1] - df['low'].iloc[-1]) / current_price * 100:.2f}%")

    # المؤشرات الفنية
    print(colored("🔧 المؤشرات المتقدمة:", "white", attrs=['bold']))
    print(f"   📊 RSI: {rsi:.1f} | ADX: {adx:.1f} | "
          f"MACD: {indicators.get('macd_hist', 0):.6f} | "
          f"ATR: {indicators.get('atr', 0):.6f}")

    # ذكاء السوق
    print(colored("🧠 ذكاء السوق:", "white", attrs=['bold']))
    print(f"   🎯 النظام: {market_analysis['market_regime']} | "
          f"الاتجاه: {market_analysis['sentiment_score']:.2f} | "
          f"السحابة: {'صاعد' if current_price > market_analysis['cloud_top'] else 'هابط'}")

    # قرار الذكاء الاصطناعي
    print(colored("🤖 قرار الذكاء الاصطناعي:", "white", attrs=['bold']))
    print(f"   🎯 الإجراء: {colored(decision['action'], action_color)} | "
          f"الاتجاه: {decision['direction'] or 'N/A'} | "
          f"الثقة: {decision['confidence']:.1%} {confidence_level}")

    if decision['reasons']:
        print(f"   📝 الأسباب: {', '.join(decision['reasons'])}")

    # معلومات المركز
    print(colored("💼 المركز الحالي:", "white", attrs=['bold']))
    print(f"   {position_info}")

    # أداء الذكاء الاصطناعي
    print(colored("📊 أداء الذكاء الاصطناعي:", "white", attrs=['bold']))
    perf = ai_council.performance_tracker
    win_rate = (perf['profitable_trades'] / perf['total_trades'] * 100) if perf['total_trades'] > 0 else 0
    print(f"   📈 الصفقات: {perf['total_trades']} | معدل الربح: {win_rate:.1f}% | "
          f"إجمالي الربح: {perf['total_pnl']:.4f}")

    print("="*120 + "\n")
