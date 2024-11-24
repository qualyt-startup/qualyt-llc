import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  standalone: true,
  template: `
    <div style="text-align:center; margin-top:50px;">
      <h1>Welcome to My Angular App!</h1>
      <p>This is a standalone Angular application.</p>
    </div>
  `,
  styles: [
    `
      h1 {
        color: #1976d2;
      }
    `,
  ],
})
export class AppComponent {}
