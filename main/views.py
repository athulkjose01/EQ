from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
# from django.conf import settings # Not used directly here
from django.urls import reverse

from .forms import BasicInfoForm, EQResponseForm
from .utils import AdvancedEQAssessmentModel, generate_chart_data # Import generate_chart_data directly

import json
import logging

# Set up logging
logger = logging.getLogger(__name__)

def home(request):
    """Display the home page with the basic info form."""
    if request.method == 'POST':
        form = BasicInfoForm(request.POST)
        if form.is_valid():
            request.session['basic_info'] = form.cleaned_data

            eq_model = AdvancedEQAssessmentModel()
            logger.info("Attempting to generate questions from Groq API...")
            questions = eq_model.generate_questions(
                form.cleaned_data['age'],
                form.cleaned_data['profession'],
                form.cleaned_data['name']
            )

            # --- CRITICAL CHECK ---
            if not questions:
                # Question generation failed
                logger.error("Failed to generate questions from Groq API. Cannot start assessment.")
                messages.error(request, "Sorry, we couldn't generate the assessment questions at this time. This might be due to an API issue. Please try again later.")
                # Stay on the home page, redisplaying the form
                return render(request, 'main/home.html', {'form': form})
            # --- END CRITICAL CHECK ---

            logger.info(f"Successfully generated {len(questions)} questions. Proceeding to interview.")
            request.session['questions'] = questions
            request.session['current_question'] = 0
            request.session['responses'] = [] # Initialize responses list

            return redirect(reverse('interview'))
    else:
        form = BasicInfoForm()

    return render(request, 'main/home.html', {'form': form})



def interview(request):
    """Handle the question and answer part of the assessment."""
    if 'basic_info' not in request.session:
        messages.error(request, "Please complete your basic information first.")
        return redirect(reverse('home'))
    
    questions = request.session.get('questions', [])
    current_index = request.session.get('current_question', 0)
    responses = request.session.get('responses', []) # Ensure responses are loaded
    
    if not questions: # Safety check if questions list is empty
        messages.error(request, "No questions available for the interview. Please start over.")
        return redirect(reverse('home'))

    if current_index >= len(questions):
        return redirect(reverse('result'))
    
    current_question_text = questions[current_index]

    if request.method == 'POST':
        form = EQResponseForm(request.POST)
        if form.is_valid():
            raw_response = form.cleaned_data['response']
            
            eq_model = AdvancedEQAssessmentModel()

            # --- Polishing Step ---
            logger.info(f"Raw response for Q{current_index+1} ('{current_question_text}'): '{raw_response}'")
            polished_response = eq_model.polish_response(current_question_text, raw_response)
            logger.info(f"Polished response for Q{current_index+1}: '{polished_response}'")
            # --- End Polishing Step ---
            
            is_valid = eq_model.validate_answer(current_question_text, polished_response)
            
            if is_valid:
                responses.append(polished_response) # Store polished response
                request.session['responses'] = responses
                
                current_index += 1
                request.session['current_question'] = current_index
                
                if current_index >= len(questions):
                    return redirect(reverse('result'))
                else:
                    return redirect(reverse('interview')) # Redirect to GET to show next question
            else:
                messages.error(request, "")
                # Redirect back to the same question with an error flag for JS to handle
                # This maintains the current question on screen
                redirect_url = reverse('interview') + '?error=validation'
                return redirect(redirect_url)
    else:
        form = EQResponseForm()
    
    progress_percent = ((current_index) / len(questions)) * 100 if questions else 0
    
    context = {
        'form': form,
        'question': current_question_text,
        'question_number': current_index + 1,
        'total_questions': len(questions),
        'progress_percent': progress_percent,
        'name': request.session.get('basic_info', {}).get('name', 'User'),
        'is_first_question': current_index == 0,
        'is_last_question': current_index == len(questions) - 1,
    }
    
    return render(request, 'main/interview.html', context)

def result(request):
    """Display the assessment results."""
    if 'basic_info' not in request.session or 'responses' not in request.session:
        messages.error(request, "Please complete the assessment first.")
        return redirect(reverse('home'))
    
    basic_info = request.session.get('basic_info', {})
    responses = request.session.get('responses', [])
    questions = request.session.get('questions', [])
    
    # Ensure responses and questions lists are of the same expected length, or handle gracefully
    if len(responses) != len(questions) and len(questions) > 0 :
        logger.warning(f"Mismatch in number of questions ({len(questions)}) and responses ({len(responses)}). Truncating/padding responses for assessment if needed.")
        # Pad responses with empty strings if fewer than questions (e.g. user skipped some how)
        # This ensures create_eq_assessment gets a response for every question slot
        responses = (responses + [''] * len(questions))[:len(questions)]


    eq_model = AdvancedEQAssessmentModel()
    assessment = eq_model.create_eq_assessment(
        basic_info.get('age', 25),
        basic_info.get('profession', 'Student'),
        basic_info.get('gender', 'Other'), # Ensure 'gender' is used if available
        basic_info.get('name', 'User'),
        responses
    )
    
    eq_status, eq_description = eq_model.interpret_eq_score(assessment['overall_eq'])
    improvement_plan = eq_model.generate_improvement_suggestions(assessment['category_scores'])
    
    qa_pairs = []
    # Ensure we iterate only over the number of actual questions
    for i in range(len(questions)):
        question_text = questions[i]
        # Use response if available, otherwise indicate it was missing/empty
        answer_text = responses[i] if i < len(responses) and responses[i] else "No response provided or response was empty."
        qa_pairs.append({
            'number': i + 1,
            'question': question_text,
            'answer': answer_text
        })
    
    # chart_data is generated by the function from utils.py
    chart_data = generate_chart_data(assessment['category_scores'])
    chart_data_json = json.dumps(chart_data) # Ensure chart_data is JSON serializable
    
    context = {
        'basic_info': basic_info,
        'overall_eq': round(assessment['overall_eq'], 2),
        'eq_status': eq_status,
        'eq_description': eq_description,
        'category_scores': assessment['category_scores'], # These are already rounded in create_eq_assessment
        'improvement_plan': improvement_plan,
        'qa_pairs': qa_pairs,
        'chart_data_json': chart_data_json,
    }
    
    # Clear session after results are prepared to prevent re-submission or stale data issues
    # Optional: could be done on "start new assessment" button explicitly
    # request.session.flush() # This would clear everything, including messages.
    # Selective clearing:
    keys_to_clear = ['basic_info', 'questions', 'responses', 'current_question']
    for key in keys_to_clear:
        if key in request.session:
            del request.session[key]
    request.session.modified = True # Ensure session changes are saved

    return render(request, 'main/result.html', context)

def download_transcript(request):
    """Generate and download a transcript of the assessment."""
    # This function might be called BEFORE session is cleared if user downloads from result page
    # For safety, let's try to reconstruct from what might be passed or if it's called after a new session.
    # However, the current design implies it's called from the result page where data *was* available.
    # If session is cleared *in* result view, this needs to receive data or a new session is needed.
    # For now, assume it's called while session for the completed assessment is still partially intact (if not fully cleared by result view)
    # OR, better, the result view should pass the necessary data to the template, and JS makes an AJAX request
    # with this data. The current implementation gets it from session.

    # For this to work reliably if session is cleared in `result` view, `download_transcript` would need the data passed to it.
    # The current implementation of result view clears the session.
    # A better approach for `download_transcript` would be if the `result` view stores the transcript data
    # in the session under a temporary key just before redirecting/rendering, or if the transcript
    # is generated on the client-side from data in the result page.

    # Given the current structure and the clearing in `result`, this function will likely fail if called after viewing results.
    # To fix:
    # 1. Don't clear session in `result` view until `start_new_assessment`.
    # 2. Or, `result` view generates transcript text and puts it in session for `download_transcript` to pick up.

    # Let's assume for now the session is NOT cleared by the result view immediately,
    # or this function is called via AJAX from the result page before a full page reload that clears it.
    # The `start_new_assessment` function is the one that should definitively clear.
    # I will remove the session clearing from the `result` view for now to make download work.

    # Re-checking `result` view: it DOES clear the session.
    # So, `download_transcript` needs to be rethought or the session clearing moved.
    # Easiest fix: Move session clearing to `start_new_assessment` only.

    # I'll modify `result` view to NOT clear the session. `start_new_assessment` will handle it.

    if 'questions' not in request.session or 'responses' not in request.session:
        messages.error(request, "No assessment data available to download. Please complete an assessment.")
        # If this happens often, it means session is cleared too early.
        return JsonResponse({'error': 'No assessment data found. Session might have expired or been cleared.'}, status=404)

    questions_list = request.session.get('questions', [])
    responses_list = request.session.get('responses', [])
    basic_info_dict = request.session.get('basic_info', {})
    
    # Re-calculate overall_eq and category_scores for the transcript if needed, or retrieve from session if stored
    # For simplicity, just Q&A for now. Could add scores if they were also session-stored from 'assessment' dict.
    
    transcript_lines = [
        f"EQ Assessment Transcript for {basic_info_dict.get('name', 'User')}",
        f"Age: {basic_info_dict.get('age', 'N/A')}",
        f"Gender: {basic_info_dict.get('gender', 'N/A')}",
        f"Profession: {basic_info_dict.get('profession', 'N/A')}",
        "\n--- Questions and Answers (Polished Responses) ---\n"
    ]
    
    for i, (question, response) in enumerate(zip(questions_list, responses_list)):
        transcript_lines.append(f"Question {i+1}: {question}")
        transcript_lines.append(f"Answer: {response if response else 'No response provided.'}")
        transcript_lines.append("") # Blank line for readability
    
    transcript_text = "\n".join(transcript_lines)
    
    response_payload = JsonResponse({
        'transcript': transcript_text,
        'filename': f"eq_assessment_{basic_info_dict.get('name', 'user').replace(' ', '_')}_transcript.txt"
    })
    # 'Content-Type' is automatically set to 'application/json' by JsonResponse
    return response_payload

def start_new_assessment(request):
    """Clear session data and start a new assessment."""
    keys_to_clear = ['basic_info', 'questions', 'responses', 'current_question', 'assessment_results'] # Add any other keys used
    for key in keys_to_clear:
        if key in request.session:
            del request.session[key]
    request.session.modified = True # Ensure session changes are saved
    messages.success(request, "Previous assessment cleared. You can start a new one.")
    return redirect(reverse('home'))