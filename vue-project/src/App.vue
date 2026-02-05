<script>
export default {
    data() {
        return {
            // بيانات مخطط المشاعر
            sentimentData: {
                labels: ['إيجابي', 'سلبي', 'محايد'],
                datasets: [{
                    data: [65.4, 17.66, 16.91],
                    backgroundColor: ['#10b981', '#ef4444', '#f59e0b'],
                    borderWidth: 3,
                    borderColor: '#fff'
                }]
            },

            // بيانات الشكاوى
            complaintsData: {
                labels: ['الشرائح', 'الاتصالات', 'المندوب', 'الشركات'],
                datasets: [{
                    label: 'عدد التكرارات',
                    data: [37, 20, 18, 5],
                    backgroundColor: ['#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6'],
                    borderRadius: 8
                }]
            },

            // بيانات المواضيع
            topicsData: {
                labels: ['نقص الشرائح', 'مشكلات التطبيق', 'استجابة المندوبين', 'قنوات التوريد', 'المسابقات', 'مواضيع أخرى'],
                datasets: [{
                    data: [31, 16, 15, 5, 7, 26],
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.7)',
                        'rgba(245, 158, 11, 0.7)',
                        'rgba(59, 130, 246, 0.7)',
                        'rgba(139, 92, 246, 0.7)',
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(107, 114, 128, 0.7)'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },

            // بيانات الرؤى
            insightsData: {
                labels: ['نقص الشرائح', 'عدم استجابة المندوبين', 'مشكلات التطبيق', 'قنوات توريد واحدة', 'تأخر إعلانات المسابقات'],
                datasets: [
                    {
                        label: 'التكرار (%)',
                        data: [31, 15, 16, 5, 7],
                        backgroundColor: 'rgba(239, 68, 68, 0.2)',
                        borderColor: '#ef4444',
                        borderWidth: 2,
                        pointBackgroundColor: '#ef4444',
                        pointRadius: 5
                    },
                    {
                        label: 'الأثر التجاري (تقديري)',
                        data: [90, 75, 70, 85, 45],
                        backgroundColor: 'rgba(59, 130, 246, 0.2)',
                        borderColor: '#3b82f6',
                        borderWidth: 2,
                        pointBackgroundColor: '#3b82f6',
                        pointRadius: 5
                    }
                ]
            },

            // بيانات الجدول الزمني
            timelineData: {
                labels: ['سياسة العمولات', 'نظام التتبع', 'إعلان المسابقات', 'استقرار التطبيق', 'مخزون الشرائح', 'تنويع الموردين'],
                datasets: [{
                    label: 'عدد الأسابيع',
                    data: [3, 4, 4, 6, 10, 14],
                    backgroundColor: [
                        '#10b981',
                        '#10b981',
                        '#f59e0b',
                        '#f59e0b',
                        '#ef4444',
                        '#ef4444'
                    ],
                    borderRadius: 8
                }]
            },

            // بيانات مؤشرات الأداء
            kpiData: {
                labels: ['توافر الشرائح', 'استجابة المندوب', 'استقرار التطبيق', 'دفع العمولات', 'مشاركة المسابقات', 'CSAT العام'],
                datasets: [{
                    label: 'الهدف المستهدف (%)',
                    data: [95, 100, 97, 100, 120, 80],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },

            chartData: {
                labels: ['January', 'February', 'March', 'April', 'May', 'June', 'July'],
                datasets: [
                    {
                        type: 'line',
                        label: 'Dataset 1',
                       borderColor: '#10b981',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        data: [50, 25, 12, 48, 56, 76, 42]
                    },
                    {
                        type: 'bar',
                        label: 'Dataset 2',
                       borderColor: '#10b981',
                        data: [21, 84, 24, 75, 37, 65, 34],
                        borderColor: 'white',
                        borderWidth: 2
                    },
                    {
                        type: 'bar',
                        label: 'Dataset 3',
                       borderColor: '#10b981',
                        data: [41, 52, 24, 74, 23, 21, 32]
                    }
                ]
            },


            // خيارات عامة
            chartOptions: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 15,
                            font: { size: 14 }
                        }
                    }
                }
            },

            // خيارات الأعمدة الأفقية
            horizontalBarOptions: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        grid: { color: 'rgba(0,0,0,0.05)' }
                    }
                }
            },

            // خيارات الرادار
            radarOptions: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom' }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            },

            // خيارات الأعمدة العمودية
            barOptions: {
                responsive: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'الأسابيع'
                        }
                    }
                }
            },

            // خيارات الخط
            lineOptions: {
                responsive: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 130,
                        title: {
                            display: true,
                            text: 'النسبة المئوية (%)'
                        }
                    }
                }
            },

            chartOptions: {
                maintainAspectRatio: false,
                aspectRatio: 0.6,
                plugins: {
                    legend: {
                        labels: {
                            color: '#10b981'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#10b981'
                        },
                        grid: {
                            color: '#10b981'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#10b981'
                        },
                        grid: {
                            color: '#10b981'
                        }
                    }
                }
    
            }
            
        }
    }
}
</script>

<template>
    <div class="container">
        <h1>📊 لوحة تحليل رضا التجار</h1>
        <p class="subtitle">تحليل شامل لاستطلاع CSAT - تجار مع مندوب مبيعات</p>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">674</div>
                <div class="stat-label">إجمالي الردود</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">65.4%</div>
                <div class="stat-label">ردود إيجابية</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">17.7%</div>
                <div class="stat-label">ردود سلبية</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">31%</div>
                <div class="stat-label">شكاوى الشرائح</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <h3 class="chart-title">توزيع المشاعر العامة</h3>
                <Chart type="doughnut" :data="sentimentData" :options="chartOptions" />
            </div>

            <div class="chart-container">
                <h3 class="chart-title">أبرز الشكاوى (حسب التكرار)</h3>
                <Chart type="bar" :data="complaintsData" :options="horizontalBarOptions" />
            </div>

            <div class="chart-container full-width">
                <h3 class="chart-title">توزيع المواضيع الرئيسية</h3>
                <Chart type="polarArea" :data="topicsData" :options="chartOptions" />
            </div>

            <div class="chart-container full-width">
                <h3 class="chart-title">الرؤى الرئيسية والأثر التجاري</h3>
                <Chart type="radar" :data="insightsData" :options="radarOptions" />
            </div>

            <div class="chart-container">
                <h3 class="chart-title">الأولويات التنفيذية (حسب الإطار الزمني)</h3>
                <Chart type="bar" :data="timelineData" :options="barOptions" />
            </div>


            <div class="chart-container">
                <h3 class="chart-title">مؤشرات الأداء المستهدفة (KPIs)</h3>
                <Chart type="line" :data="kpiData" :options="lineOptions" />
            </div>

            <div class="chart-container">
                <h3 class="chart-title">مؤشرات الأداء المستهدفة (KPIs)</h3>
                <Chart type="bar" :data="chartData" :options="chartOptions" class="h-[30rem]" />
            </div>
        </div>
    </div>
</template>

<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        min-height: 100vh;
    }

    .container {
        max-width: 1400px;
        margin: 0 auto;
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }

    h1 {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 10px;
        font-size: 2.5em;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 40px;
        font-size: 1.1em;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 40px;
    }

    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-5px);
    }

    .stat-number {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .stat-label {
        font-size: 0.9em;
        opacity: 0.9;
    }

    .charts-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
        gap: 30px;
        margin-bottom: 30px;
    }

    .chart-container {
        background: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .chart-title {
        color: #2c3e50;
        margin-bottom: 20px;
        font-size: 1.3em;
        font-weight: bold;
        text-align: center;
    }

    .full-width {
        grid-column: 1 / -1;
    }

    @media (max-width: 768px) {
        .charts-grid {
            grid-template-columns: 1fr;
        }

        h1 {
            font-size: 1.8em;
        }

        .stat-number {
            font-size: 2em;
        }
    }
</style>