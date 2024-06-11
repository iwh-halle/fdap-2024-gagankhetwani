from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
import requests
import time
from .forms import SnipForm
from .models import Snip
import random
from django.db.models import Count
from django.db.models.functions import TruncDate

def home(request):
    message = None
    initial_data = {}
    snip_id = request.GET.get('snip_id')
    if snip_id:
        initial_data['snip_id'] = snip_id
    if request.method == 'POST':
        form = SnipForm(request.POST)
        if form.is_valid():
            snip_id = form.cleaned_data['snip_id']
            student_id = form.cleaned_data['student_id']
            try:
                snip = Snip.objects.get(snip_id=snip_id)
                snip.claim_attempts += 1
                if snip.student_id:
                    message = 'Nice try, but this Snip has already been claimed. 😑'
                    form = SnipForm()
                else:
                    snip.student_id = student_id
                    classroom = snip.snipsheet.classroom
                    previously_claimed_snips = Snip.objects.filter(snipsheet__classroom=classroom, student_id=student_id).count()
                    emoji = random.choice(['🎉', '👏', '🎈', '🥳', '😎', '🙌'])
                    message = f'Awesome, you have successfully claimed your {ordinal_format(previously_claimed_snips + 1)} Snip! {emoji}'
                    form = SnipForm()
                snip.save()
            except Snip.DoesNotExist:
                message = 'Sorry, this Snip does not exist. ☹️'
    else:
        form = SnipForm(initial=initial_data)
    context = {'form': form, 'message': message}
    return render(request, 'snips/home.html', context)

def api_chart_data(request):
    data = (Snip.objects
                .filter(student_id__isnull=False, student_id__gt='')
                .annotate(date=TruncDate('updated_at'))
                .values('date')
                .annotate(count=Count('id'))
                .order_by('date'))
    chart_data = [{'date': entry['date'].isoformat(), 'count': entry['count']} for entry in data]
    return JsonResponse(chart_data, safe=False)

def deribit_data(request):
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    instrument_name = "BTC-PERPETUAL"
    resolution = "60"
    current_time = int(time.time() * 1000)
    start_time = current_time - (1 * 365 * 24 * 60 * 60 * 1000)
    
    params = {
        "instrument_name": instrument_name,
        "resolution": resolution,
        "start_timestamp": start_time,
        "end_timestamp": current_time,
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()['result']
        return JsonResponse(data, safe=False)
    else:
        return JsonResponse({'error': 'Failed to fetch data'}, status=response.status_code)

def chart_page(request):
    return render(request, 'snips/chart.html')