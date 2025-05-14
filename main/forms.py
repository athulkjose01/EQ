from django import forms

class BasicInfoForm(forms.Form):
    name = forms.CharField(label='Your Name', max_length=100, 
                          widget=forms.TextInput(attrs={'class': 'form-control'}))
    age = forms.IntegerField(label='Age', min_value=10, max_value=100, initial=25,
                            widget=forms.NumberInput(attrs={'class': 'form-control'}))
    gender = forms.ChoiceField(
        label='Gender',
        choices=[('Male', 'Male'), ('Female', 'Female'), ('Non-Binary', 'Non-Binary')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    profession = forms.ChoiceField(
        label='Profession',
        choices=[
            ('Student', 'Student'),
            ('Corporate', 'Corporate'),
            ('Healthcare', 'Healthcare'),
            ('Creative', 'Creative'),
            ('Entrepreneur', 'Entrepreneur')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )

class EQResponseForm(forms.Form):
    response = forms.CharField(
        label='Your Answer',
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 5,
            'placeholder': 'Please provide your answer here...'
        }),
        required=True
    )